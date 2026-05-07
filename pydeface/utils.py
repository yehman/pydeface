"""Utility scripts for pydeface."""

import os
import shutil
import sys
import tempfile
from importlib.resources import files

import numpy as np
from nibabel import Nifti1Image, load
from nipype.interfaces import fsl


def initial_checks(template=None, facemask=None, template_brain=None):
    """Initial sanity checks."""
    if template is None:
        template = files('pydeface').joinpath('data/mean_reg2mean.nii.gz')
    if facemask is None:
        facemask = files('pydeface').joinpath('data/facemask.nii.gz')
    if template_brain is None:
        template_brain = files('pydeface').joinpath('data/MNI_brain_padded.nii.gz')

    if not os.path.exists(template):
        raise Exception(f'Missing template: {template}')
    if not os.path.exists(facemask):
        raise Exception(f'Missing face mask: {facemask}')
    if not os.path.exists(template_brain):
        raise Exception(f'Missing skull-stripped template: {template_brain}')

    if 'FSLDIR' not in os.environ:
        raise Exception(
            'FSL must be installed and FSLDIR environment variable must be defined.'
        )
        sys.exit(2)
    return template, facemask, template_brain


def output_checks(infile, outfile=None, force=False):
    """Determine output file name."""
    if force is None:
        force = False
    if outfile is None:
        outfile = infile.replace('.nii', '_defaced.nii')

    if os.path.exists(outfile) and force:
        print('Previous output will be overwritten.')
    elif os.path.exists(outfile):
        raise Exception(
            f"{outfile} already exists. Remove it first or use '--force' "
            'flag to overwrite.'
        )
    else:
        pass
    return outfile


def generate_tmpfiles(verbose=True):
    _, template_reg_mat = tempfile.mkstemp(suffix='.mat')
    _, warped_mask = tempfile.mkstemp(suffix='.nii.gz')
    if verbose:
        print(f'Temporary files:\n  {template_reg_mat}\n  {warped_mask}')
    _, template_reg = tempfile.mkstemp(suffix='.nii.gz')
    _, warped_mask_mat = tempfile.mkstemp(suffix='.mat')
    return template_reg, template_reg_mat, warped_mask, warped_mask_mat


def cleanup_files(*args):
    print('Cleaning up...')
    for p in args:
        if os.path.exists(p):
            os.remove(p)


def get_outfile_type(outpath):
    # Returns fsl output type for passing to fsl's flirt
    if outpath.endswith('nii.gz'):
        return 'NIFTI_GZ'
    elif outpath.endswith('nii'):
        return 'NIFTI'
    else:
        raise ValueError('outfile path should be have .nii or .nii.gz suffix')


def compute_nmi(img1_path, img2_path, bins=20):
    """Compute Normalized Mutual Information (NMI) between two NIfTI images to check alignment."""
    data1 = load(img1_path).get_fdata().flatten()
    data2 = load(img2_path).get_fdata().flatten()

    # Filter out background to focus on brain structure
    mask = (data1 > np.percentile(data1, 10)) & (data2 > np.percentile(data2, 10))
    if np.sum(mask) == 0:
        return 0.0

    d1, d2 = data1[mask], data2[mask]

    # Calculate 2D histogram
    hist_2d, _, _ = np.histogram2d(d1, d2, bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

    if hx + hy == 0:
        return 0.0

    return 2.0 * mi / (hx + hy)


def deface_image(
    infile=None,
    outfile=None,
    facemask=None,
    template=None,
    cost='mutualinfo',
    force=False,
    forcecleanup=False,
    verbose=True,
    qc_threshold=0.2,
    template_brain=None,
    bet=False,
    auto_bet=False,
    **kwargs,
):
    if not infile:
        raise ValueError('infile must be specified')
    if shutil.which('fsl') is None:
        raise OSError('fsl cannot be found on the path')

    template, facemask, template_brain = initial_checks(template, facemask, template_brain)
    outfile = output_checks(infile, outfile, force)
    template_reg, template_reg_mat, warped_mask, warped_mask_mat = generate_tmpfiles()

    print(f'Defacing...\n  {infile}')

    do_fallback = bet
    outfile_type = get_outfile_type(template_reg)

    # register template to infile if user did not use --bet
    if not bet:
        flirt = fsl.FLIRT()
        flirt.inputs.cost_func = cost
        flirt.inputs.in_file = template
        flirt.inputs.out_matrix_file = template_reg_mat
        flirt.inputs.out_file = template_reg
        flirt.inputs.output_type = outfile_type
        flirt.inputs.reference = infile
        flirt.run()

        # Run QC check for --auto-bet option
        if auto_bet:
            nmi_score = compute_nmi(infile, template_reg)
            if verbose:
                print(f"Registration QC (NMI score): {nmi_score:.3f}")

            if nmi_score < qc_threshold:
                print(f"Warning: Registration NMI score ({nmi_score:.3f}) is below threshold ({qc_threshold}).")
                do_fallback = True
            else:
                if verbose:
                    print("Registration passed QC.")

    # If the QC failed OR the user manually passed --bet, run the robust skull-stripped registration
    if do_fallback:
        if bet:
            if verbose:
                print("Manual --bet flag provided. Bypassing standard registration...")
        else:
            print("Falling back to skull-stripped registration...")

        # Define paths for the temporary skull-stripped subject
        infile_brain = infile.replace('.nii', '_brain.nii')
        if '.gz' not in infile_brain and infile.endswith('.gz'):
            infile_brain += '.gz'

        # Run FSL BET to skull-strip the input image
        bet_node = fsl.BET()
        bet_node.inputs.in_file = infile
        bet_node.inputs.out_file = infile_brain
        bet_node.inputs.output_type = get_outfile_type(infile_brain)
        bet_node.run()

        # Run FLIRT using the skull-stripped brain template to the subject brain
        flirt = fsl.FLIRT()
        flirt.inputs.cost_func = cost
        flirt.inputs.in_file = template_brain
        flirt.inputs.out_matrix_file = template_reg_mat
        flirt.inputs.out_file = template_reg
        flirt.inputs.output_type = outfile_type
        flirt.inputs.reference = infile_brain
        flirt.run()

        # Clean up the temporary brain file
        if os.path.exists(infile_brain):
            os.remove(infile_brain)

    outfile_type = get_outfile_type(warped_mask)
    # warp facemask to infile
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = facemask
    flirt.inputs.in_matrix_file = template_reg_mat
    flirt.inputs.apply_xfm = True
    flirt.inputs.reference = infile
    flirt.inputs.out_file = warped_mask
    flirt.inputs.output_type = outfile_type
    flirt.inputs.out_matrix_file = warped_mask_mat
    flirt.run()

    # multiply mask by infile and save
    infile_img = load(infile)
    infile_data = np.asarray(infile_img.dataobj)
    warped_mask_img = load(warped_mask)
    warped_mask_data = np.asarray(warped_mask_img.dataobj)
    try:
        outdata = infile_data.squeeze() * warped_mask_data
    except ValueError:
        tmpdata = np.stack(warped_mask_data * infile_img.shape[-1], axis=-1)
        outdata = infile_data * tmpdata

    masked_brain = Nifti1Image(outdata, infile_img.affine, infile_img.header)
    masked_brain.to_filename(outfile)
    print(f'Defaced image saved as:\n  {outfile}')

    if forcecleanup:
        cleanup_files(warped_mask, template_reg, template_reg_mat)
        return warped_mask_img
    else:
        return warped_mask_img, warped_mask, template_reg, template_reg_mat


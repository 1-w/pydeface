"""Utility scripts for pydeface."""

import os
import shutil
import sys
from numpy.core.numeric import _outer_dispatcher
from pkg_resources import resource_filename, Requirement
import tempfile
import nipype.interfaces.fsl as fsl
from pathlib import Path

from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import Function

def initial_checks(template=None, facemask=None):
    """Initial sanity checks."""
    if template is None:
        template = resource_filename(Requirement.parse("pydeface"),
                                     "pydeface/data/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz")
    if facemask is None:
        facemask = resource_filename(Requirement.parse("pydeface"),
                                     "pydeface/data/mni_icbm152_t1_tal_nlin_asym_09a_face_mask_filled_resampled_resized.nii.gz")

    if not os.path.exists(template):
        raise Exception('Missing template: %s' % template)
    if not os.path.exists(facemask):
        raise Exception('Missing face mask: %s' % facemask)

    if 'FSLDIR' not in os.environ:
        raise Exception("FSL must be installed and "
                        "FSLDIR environment variable must be defined.")
        sys.exit(2)
    return template, facemask


def output_checks(infile, outfile=None, force=False):
    """Determine output file name."""
    if force is None:
        force = False
    if outfile is None:
        outfile = infile.replace('.nii', '_defaced.nii')

    if os.path.exists(outfile) and force:
        print('Previous output will be overwritten.')
    elif os.path.exists(outfile):
        raise Exception("%s already exists. Remove it first or use '--force' "
                        "flag to overwrite." % outfile)
    else:
        pass
    return outfile

# def generate_tmpfiles(verbose=True):
#     _, template_reg_mat = tempfile.mkstemp(suffix='.mat')
#     _, warped_mask = tempfile.mkstemp(suffix='.nii.gz')
#     if verbose:
#         print("Temporary files:\n  %s\n  %s" % (template_reg_mat, warped_mask))
#     _, template_reg = tempfile.mkstemp(suffix='.nii.gz')
#     _, warped_mask_mat = tempfile.mkstemp(suffix='.mat')
#     _, resize_mat = tempfile.mkstemp(suffix='.mat')
#     _, combined_mat = tempfile.mkstemp(suffix='.mat')
#     return template_reg, template_reg_mat, warped_mask, warped_mask_mat, resize_mat, combined_mat


def cleanup_files(*args):
    print("Cleaning up...")
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

def save_img(in_file,target_dir=''):
    from nibabel import load, Nifti1Image
    import numpy as np
    from PIL import Image
    import os

    infile_img = load(in_file).get_fdata()

    shape = infile_img.shape

    mid = int(shape[0]/2)

    img = Image.fromarray(infile_img[mid,:,:])
    img = img.convert("L")

    if target_dir == '':
        target_dir = os.path.join(os.path.split(in_file)[0],'control_images')

    os.makedirs(target_dir,exist_ok=True)
    outfiledir = target_dir
    outfilename = os.path.basename(in_file).split('.')[0] + '_control.png'
    img.save(os.path.join(outfiledir,outfilename))

def removeMask(in_file, mask, outfile):
        from nibabel import load, Nifti1Image
        import numpy as np

        # multiply mask by infile and save
        infile_img = load(in_file)
        warped_mask_img = load(mask)

        #invert mask
        warped_mask_img_data = -(warped_mask_img.get_data()-1)

        try:
            outdata = infile_img.get_data().squeeze() * warped_mask_img_data
        except ValueError:
            tmpdata = np.stack([warped_mask_img_data] *
                            infile_img.get_data().shape[-1], axis=-1)
            outdata = infile_img.get_data() * tmpdata

        masked_brain = Nifti1Image(outdata, infile_img.get_affine(),
                                infile_img.get_header())
        if outfile.endswith("nii"):
            outfile += ".gz"
        masked_brain.to_filename(outfile)
        return outfile

def deface_image(infile=None, outfile=None, facemask=None,
                 template=None, cost='mutualinfo', force=False,
                 forcecleanup=False, verbose=True, cwd=None, **kwargs):
    if not infile:
        raise ValueError("infile must be specified")
    if shutil.which('fsl') is None:
        raise EnvironmentError("fsl cannot be found on the path")

    template, facemask = initial_checks(template, facemask)
    outfile = output_checks(infile, outfile, force)

    templates = {'template': template,\
    'facemask': facemask,\
    'inputImg' : infile}

    defaceWf = Workflow(name='defaceWf')
    if cwd is not None:
        defaceWf.base_dir = cwd
        print('set cwd to',defaceWf.base_dir)
        print(os.path.normpath(os.path.join(cwd,outfile)))

    if not os.path.isabs(infile):
        infile = os.path.normpath(os.path.join(cwd, infile))

    if not os.path.isabs(outfile):
        outfile = os.path.normpath(os.path.join(cwd, outfile))

    selectfiles = Node(SelectFiles(templates),name="selectfiles")
    selectfiles.inputs.base_directory = defaceWf.base_dir

    flirtT2Img = Node(fsl.FLIRT(cost_func = cost,dof = 12),name='flirtT2T1')
    defaceWf.connect(selectfiles,'inputImg',flirtT2Img,'reference')
    defaceWf.connect(selectfiles,'template',flirtT2Img,'in_file')

    ApplyXfmRef2Mask = Node(fsl.preprocess.ApplyXFM(uses_qform=True, no_search=True, apply_xfm=True),name="ApplyXfmRef2Mask")
    defaceWf.connect(selectfiles,'facemask',ApplyXfmRef2Mask,'in_file')
    defaceWf.connect(selectfiles,'template',ApplyXfmRef2Mask,'reference')

    convertXfm = Node(fsl.ConvertXFM(concat_xfm=True),name='convertXfm')
    defaceWf.connect(ApplyXfmRef2Mask,'out_matrix_file',convertXfm,'in_file')
    defaceWf.connect(flirtT2Img,'out_matrix_file',convertXfm,'in_file2')
    
    ApplyXfmMask2Img = Node(fsl.preprocess.ApplyXFM(apply_xfm=True),name="ApplyXfmMask2Img")
    defaceWf.connect(selectfiles,'facemask',ApplyXfmMask2Img,'in_file')
    defaceWf.connect(convertXfm,'out_file',ApplyXfmMask2Img,'in_matrix_file')
    defaceWf.connect(selectfiles,'inputImg',ApplyXfmMask2Img,'reference')

    

    removeMaskNode = Node(name='removeMask',
               interface=Function(input_names=['in_file', 'mask', 'outfile'],
                                  output_names=['out_file'],
                                  function=removeMask))
    removeMaskNode.inputs.outfile = outfile
    defaceWf.connect(selectfiles,'inputImg',removeMaskNode,'in_file')
    defaceWf.connect(ApplyXfmMask2Img,'out_file',removeMaskNode,'mask')

    saveControlImgNode = Node(name='saveControlImg',
               interface=Function(input_names=['in_file', 'target_dir'],
                                  output_names=[],
                                  function=save_img))
    defaceWf.connect(removeMaskNode,'out_file',saveControlImgNode,'in_file')

    #datasink = Node(DataSink(),name='sink')
    #defaceWf.connect([(selectfiles,'facemask',ApplyXfmMask2Img,'in_file'),\

    print('Defacing...\n  %s' % infile)
    defaceWf.run()

    print("Defaced image saved as:\n  %s" % outfile)


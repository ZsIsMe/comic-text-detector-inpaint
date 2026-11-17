import unittest
import numpy as np
from detect_solid_inpaint_folder import _compose_overlay_preview
from solid_inpaint_ui import _mask_overlay_image, _overlay_mask_on_bgr, _overlay_transparent_mask_on_bgr

class PreviewEquivalenceTests(unittest.TestCase):
    def test_all_opacities_match_float_compositor(self):
        rng=np.random.default_rng(24)
        base=rng.integers(0,256,(16,256,3),dtype=np.uint8)
        overlay=rng.integers(0,256,(16,256,4),dtype=np.uint8)
        overlay[:,:,3]=np.arange(256,dtype=np.uint8)
        a=overlay[:,:,3:4].astype(np.float32)/255.0
        expected=(base.astype(np.float32)*(1-a)+overlay[:,:,:3].astype(np.float32)*a).astype(np.uint8)
        np.testing.assert_array_equal(_compose_overlay_preview(base,overlay),expected)
    def test_display_blending_preserves_rounding_for_all_values(self):
        base=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(3,1,3))
        mask=np.zeros(base.shape[:2],np.uint8);mask[:,::2]=255
        active=mask>0;color=(255,113,0)
        for alpha in (0,.01,.38,.65,1):
            expected=(base.astype(np.float32)*(1-alpha)).astype(np.uint8)
            blend=(base[active].astype(np.float32)*(1-alpha)+np.array(color,dtype=np.float32)*alpha).astype(np.uint8)
            expected[active]=blend
            np.testing.assert_array_equal(_mask_overlay_image(base,mask,alpha,color),expected)
            expected=base.copy();expected[active]=blend
            for fn in (_overlay_mask_on_bgr,_overlay_transparent_mask_on_bgr):
                np.testing.assert_array_equal(fn(base,mask,alpha,color),expected)

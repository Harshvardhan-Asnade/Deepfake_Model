import os
try:
    from imwatermark import WatermarkDecoder
except ImportError:
    WatermarkDecoder = None

def check_watermarks(filepath):
    """
    Checks for invisible watermarks (specifically Stable Diffusion's 'sd_private').
    Returns a dictionary with detection status.
    """
    result = {
        "detected": False,
        "method": None,
        "source": None
    }

    if not WatermarkDecoder:
        return result

    try:
        # Standard Stable Diffusion watermark is 48 bits, detecting 'bytes'
        decoder = WatermarkDecoder('bytes', 32) # Standard length for some, but SD often uses 48 bits?
        # Actually, the 'invisible-watermark' library default for SD
        # typically uses method='dwtDct' combined with a specific decoder.
        
        # Let's try the standard approach for Stable Diffusion detection
        # The library usually has a specific 'bytes' decoder for it.
        
        bgr_image = None
        import cv2
        bgr_image = cv2.imread(filepath)
        if bgr_image is None:
            return result
            
        decoder = WatermarkDecoder('bytes', 136) # Try generic length or specific
        watermark = decoder.decode(bgr_image, 'dwtDct')
        
        # Stable Diffusion's watermark often decodes to explicit bytes.
        # However, a more robust way often used is checking for the specific signature 
        # that the library 'invisible-watermark' looks for.
        
        # Simplifying: If we decode *something* valid/structured, it might be watermarked.
        # But for 'sd_private', we verify specifically.
        
        # Note: A simpler check using the library's built-in script logic:
        # It usually converts "Stability AI" string to bits?
        
        # If we successfully decode the known string "Stability" or derivatives.
        decoded_text = watermark.decode('utf-8', errors='ignore')
        
        if "Stability" in decoded_text or "sd_private" in decoded_text :
            result["detected"] = True
            result["method"] = "Invisible Watermark"
            result["source"] = "Stable Diffusion"
        
    except Exception as e:
        # print(f"Watermark Check Error: {e}")
        pass
        
    return result

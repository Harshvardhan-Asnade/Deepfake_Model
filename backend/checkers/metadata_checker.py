import os
import json
import exifread
try:
    import c2pa
except ImportError:
    c2pa = None

def check_metadata(filepath):
    """
    Checks for Content Credentials (C2PA) and specific AI-generation metadata in Exif/XMP.
    Returns a dictionary with detection status and details.
    """
    result = {
        "detected": False,
        "method": None,
        "source": None,
        "details": {}
    }

    # 1. Check C2PA / Content Credentials
    if c2pa:
        try:
            # Correct API usage for c2pa-python
            reader = c2pa.Reader(filepath)
            manifest_json = reader.json()
            
            # Robust check: Convert to string and search for keywords
            # This avoids dependency on exact JSON structure which might vary
            json_str = manifest_json.lower()
            
            if "dall-e" in json_str:
                result["detected"] = True
                result["method"] = "C2PA"
                result["source"] = "DALL-E"
                return result
            if "adobe firefly" in json_str:
                result["detected"] = True
                result["method"] = "C2PA"
                result["source"] = "Adobe Firefly"
                return result
            if "bing image creator" in json_str:
                result["detected"] = True
                result["method"] = "C2PA"
                result["source"] = "Bing Image Creator"
                return result
            if "my-tool" in json_str: # Example placeholder
                 pass
                 
            # General check for "artificial" or "created" actions if no specific tool found
            if 'c2pa.actions' in json_str and 'artificial' in json_str:
                result["detected"] = True
                result["method"] = "C2PA"
                result["source"] = "AI Generated (C2PA)"
                return result

        except Exception as e:
            # Expected if no C2PA manifest exists
            # print(f"C2PA Check Info: {e}")
            pass

    # 2. Check Exif/XMP via ExifRead
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Common AI signatures in Exif/XMP/IPTC
            software_tags = [str(tags.get('Image Software', '')), str(tags.get('0th Software', ''))]
            description_tags = [str(tags.get('Image ImageDescription', '')), str(tags.get('EXIF UserComment', ''))]
            
            # DALL-E 3 often leaves signature in ImageDescription or Software
            for tag in software_tags + description_tags:
                tag_lower = tag.lower()
                if "dall-e" in tag_lower:
                    result["detected"] = True
                    result["method"] = "EXIF"
                    result["source"] = "DALL-E"
                    return result
                if "adobe firefly" in tag_lower:
                    result["detected"] = True
                    result["method"] = "EXIF"
                    result["source"] = "Adobe Firefly"
                    return result
                if "bing image creator" in tag_lower:
                    result["detected"] = True
                    result["method"] = "EXIF"
                    result["source"] = "Bing Image Creator"
                    return result
                if "stable diffusion" in tag_lower:
                    result["detected"] = True
                    result["method"] = "EXIF"
                    result["source"] = "Stable Diffusion"
                    return result

                # Generic check for other known AI tools based on common signatures
                for tool in ["midjourney", "runway", "leonardo", "nightcafe", "canva"]:
                    if tool in tag_lower:
                        result["detected"] = True
                        result["method"] = "EXIF"
                        result["source"] = tool.title() # Capitalize first letter
                        return result
                    
    except Exception as e:
        print(f"Exif Check Error: {e}")

    # 3. Check PNG Text Chunks (often used by Leonardo, NightCafe, Stable Diffusion)
    # ExifRead doesn't always catch purely textual PNG chunks "parameters" or "Software"
    try:
        from PIL import Image
        img = Image.open(filepath)
        img.load() # Load to access info
        
        info = img.info or {}
        
        # Combine all string values for search
        search_space = " ".join([str(v).lower() for k, v in info.items()])
        
        if "stable diffusion" in search_space:
             result["detected"] = True
             result["method"] = "PNG Metadata"
             result["source"] = "Stable Diffusion"
             return result
             
        if "midjourney" in search_space:
             result["detected"] = True
             result["method"] = "PNG Metadata"
             result["source"] = "Midjourney"
             return result

        if "leonardo" in search_space:
             result["detected"] = True
             result["method"] = "PNG Metadata"
             result["source"] = "Leonardo AI"
             return result

        if "nightcafe" in search_space:
             result["detected"] = True
             result["method"] = "PNG Metadata"
             result["source"] = "NightCafe"
             return result

        if "runway" in search_space:
             result["detected"] = True
             result["method"] = "PNG Metadata"
             result["source"] = "Runway Gen-2"
             return result

    except Exception as e:
        # print(f"PNG Check Error: {e}")
        pass


    return result

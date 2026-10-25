import ee
import datetime

# Initialize Earth Engine with your project
ee.Initialize(project='potent-cedar-475519-n5')

def get_tamil_nadu_boundary():
    """
    Retrieve Tamil Nadu boundary from administrative dataset.
    Returns an ee.Feature representing Tamil Nadu's boundary.
    """
    try:
        # Get India states dataset
        states = ee.FeatureCollection('FAO/GAUL/2015/level1')
        
        # Filter for Tamil Nadu
        tamil_nadu = states.filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu')).first()
        
        if not tamil_nadu:
            raise Exception("Tamil Nadu boundary not found in the dataset")
            
        return tamil_nadu
        
    except Exception as e:
        print(f"Error getting Tamil Nadu boundary: {str(e)}")
        raise

def mask_clouds(image):
    """
    Apply cloud masking to Sentinel-2 image.
    """
    # Get QA band
    qa = image.select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    
    # Get cloud and cirrus mask
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        
    # Apply the mask
    return image.updateMask(mask)

def get_sentinel_data(region, start_date, end_date):
    """
    Retrieve and process Sentinel-2 data for the specified region and time period.
    """
    try:
        # Get Sentinel-2 Level-2A collection (newer version)
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        # Apply cloud masking
        s2_masked = s2_collection.map(mask_clouds)
        
        # Create median composite
        median_composite = s2_masked.median()
        
        return median_composite
        
    except Exception as e:
        print(f"Error processing Sentinel-2 data: {str(e)}")
        raise

def main():
    try:
        # Get Tamil Nadu boundary
        print("Retrieving Tamil Nadu boundary...")
        tn_boundary = get_tamil_nadu_boundary()
        
        # Set time period (example: last 3 months)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=90)
        
        print("Processing Sentinel-2 data...")
        # Get Sentinel-2 data
        s2_data = get_sentinel_data(
            tn_boundary.geometry(),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Clip the image to Tamil Nadu boundary
        clipped_image = s2_data.clip(tn_boundary.geometry())
        
        # Visualization parameters for natural color composite
        viz_params = {
            'bands': ['B4', 'B3', 'B2'],  # RGB
            'min': 0,
            'max': 3000,
            'gamma': 1.2
        }
        
        # Create the export task
        task = ee.batch.Export.image.toDrive(
            image=clipped_image,
            description='Tamil_Nadu_Sentinel2_Composite',
            scale=10,  # 10m resolution
            region=tn_boundary.geometry(),
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        
        # Start the export task
        task.start()
        print("Export task started. Check your Google Drive for the results.")
        
        # Calculate and print area statistics
        area_sqkm = tn_boundary.geometry().area().divide(1e6).getInfo()
        print(f"\nArea Statistics:")
        print(f"Tamil Nadu Area: {area_sqkm:.2f} square kilometers")
        
        # Print processing metadata
        print("\nProcessing Metadata:")
        print(f"Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("Resolution: 10 meters")
        print("Dataset: Sentinel-2 Level-2A Surface Reflectance (Harmonized)")
        print("Cloud Cover Threshold: 30%")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
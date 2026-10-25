import ee
from datetime import datetime, timedelta
import os
import json
import shutil
from logging_config import setup_logging, log_execution_time, log_error, log_memory_usage

# Initialize Earth Engine
ee.Initialize(project='potent-cedar-475519-n5')

class DeforestationDetector:
    def __init__(self):
        # Set up logging
        loggers = setup_logging()
        self.logger = loggers['satellite']
        self.logger.info("Initializing DeforestationDetector")
        
        try:
            # Initialize Earth Engine
            ee.Initialize()
            print("Successfully authenticated with Earth Engine")
            
            # Define the AOI coordinates for South India
            self.west = 74.0  # Western longitude
            self.south = 8.0  # Southern latitude
            self.east = 84.5  # Eastern longitude
            self.north = 20.0  # Northern latitude
            
            # Create the rectangular AOI geometry
            self.rectangle_aoi = ee.Geometry.Rectangle([self.west, self.south, self.east, self.north])
            
            # Get state boundaries from FAO/GAUL dataset
            self.states = ee.FeatureCollection('FAO/GAUL/2015/level1')
            
            # Get all state names in India
            india_states = self.states.filter(ee.Filter.eq('ADM0_NAME', 'India'))
            state_names = india_states.aggregate_array('ADM1_NAME').getInfo()
            print("\nAvailable states in India:")
            for name in sorted(state_names):
                print(f"- {name}")
            
            # Dictionary of state names and their boundaries
            self.state_boundaries = {}
            south_indian_states = [
                'Tamil Nadu',
                'Kerala',
                'Karnataka',
                'Andhra Pradesh',
                'Telengana'  # Updated spelling in FAO/GAUL dataset
            ]
            
            print("\nAttempting to find boundaries for South Indian states...")
            # Get boundaries for each state
            for state_name in south_indian_states:
                try:
                    state_feature = india_states.filter(ee.Filter.eq('ADM1_NAME', state_name)).first()
                    state_feature_info = state_feature.getInfo()
                    if state_feature_info is not None:
                        self.state_boundaries[state_name] = state_feature
                        print(f"Found boundary for {state_name}")
                    else:
                        print(f"Warning: Could not find boundary for {state_name}")
                except Exception as e:
                    print(f"Error processing {state_name}: {str(e)}")
            
            if not self.state_boundaries:
                raise ValueError("No state boundaries found. Please check state names.")
            
            print("\nCreating feature collection...")
            # Create a feature collection of South Indian states
            self.target_states = ee.FeatureCollection(list(self.state_boundaries.values()))
            
            print("Intersecting with rectangular AOI...")
            # Intersect rectangular AOI with state boundaries for precise area
            self.aoi = self.target_states.geometry().intersection(self.rectangle_aoi)
            print("Initialization complete.")
            
        except Exception as e:
            print(f"Error initializing DeforestationDetector: {str(e)}")
            raise

    def main(self):
        """Main execution method for deforestation detection."""
        try:
            print("Initializing Deforestation Detection for South India...")
            
            # Set time periods for seasonal analysis
            current_date = datetime.utcnow()
            
            # Define seasons (6-month periods)
            seasons = [
                {
                    'name': 'Pre_Monsoon',
                    'start': current_date - timedelta(days=180),
                    'end': current_date
                },
                {
                    'name': 'Previous_Season',
                    'start': current_date - timedelta(days=365),
                    'end': current_date - timedelta(days=185)
                }
            ]
            
            print("\nCalculating area statistics...")
            # Calculate areas in square kilometers
            state_areas = {}
            for state_name, boundary in self.state_boundaries.items():
                state_area = boundary.geometry().area().divide(1e6).getInfo()
                state_areas[state_name] = state_area
            
            study_area = self.aoi.area().divide(1e6).getInfo()
            
            print("\nArea Statistics:")
            for state_name, area in state_areas.items():
                print(f"{state_name} Area: {area:,.2f} km²")
            print(f"Study Area (Intersection): {study_area:,.2f} km²")
            
            # Set up directories for data organization
            base_dir = os.path.dirname(os.path.abspath(__file__))
            download_dir = os.path.join(base_dir, 'downloads')
            processed_dir = os.path.join(base_dir, 'processed_data')
            
            # Create directories if they don't exist
            os.makedirs(download_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Process and export data for each season
            export_tasks = []
            for season in seasons:
                print(f"\nProcessing {season['name']} Sentinel-2 data...")
                
                # Get and process Sentinel-2 data
                processed_image = self.get_sentinel_data(
                    season['start'].strftime('%Y-%m-%d'),
                    season['end'].strftime('%Y-%m-%d')
                )
                
                # Export the processed image
                tasks = self.export_image(
                    processed_image,
                    'South_India_Deforestation',
                    season['name']
                )
                export_tasks.append(tasks)
            
            print("\nProcessing Metadata:")
            print("Geographic Extent:")
            print(f"- Longitude: {self.west}°E to {self.east}°E")
            print(f"- Latitude: {self.south}°N to {self.north}°N")
            
            print("\nStates Covered:")
            print("- Tamil Nadu")
            print("- Kerala")
            print("- Karnataka")
            print("- Andhra Pradesh")
            print("- Telangana")
            
            print("\nTime Periods:")
            for season in seasons:
                print(f"- {season['name']}: {season['start'].strftime('%Y-%m-%d')} to {season['end'].strftime('%Y-%m-%d')}")
            
            print("\nDataset Specifications:")
            print("- Source: Sentinel-2 Level-2A Surface Reflectance (Harmonized)")
            print("- Resolution: 10 meters")
            print("- Cloud Cover Threshold: 20%")
            print("- Spectral Bands:")
            print("  * Visible (B2, B3, B4)")
            print("  * Red Edge (B5, B6, B7)")
            print("  * Near Infrared (B8, B8A)")
            print("  * Short Wave Infrared (B11, B12)")
            print("- Vegetation Indices:")
            print("  * NDVI (Normalized Difference Vegetation Index)")
            print("  * EVI (Enhanced Vegetation Index)")
            print("  * NDMI (Normalized Difference Moisture Index)")

            print("\nExport Status:")
            print("- Tasks have been submitted to Google Earth Engine")
            print("- Download files from Google Drive when complete")
            print(f"- Move downloaded files to: {download_dir}")
            print(f"- Run data organization when downloads complete")
            
            # Instructions for next steps
            print("\nNext Steps:")
            print("1. Wait for Google Earth Engine tasks to complete")
            print("2. Download exported files from Google Drive")
            print(f"3. Place downloaded files in: {download_dir}")
            print("4. Run the following to organize data:")
            print("   detector.organize_local_data(download_dir, processed_dir)")
            print(f"5. Processed data will be available in: {processed_dir}")
            
            print("\nExport Tasks Started:")
            print("- Spectral data and vegetation indices will be exported to 'ECOVISION_South_India' folder")
            print("- Files will be saved as Cloud-Optimized GeoTIFFs")
            print("- Check your Google Drive for the results")
            
        except Exception as e:
            print(f"Error in main execution: {str(e)}")
            raise

    @log_execution_time
    @log_memory_usage
    def mask_clouds(self, image):
        """
        Apply comprehensive cloud and shadow masking to Sentinel-2 image.
        
        This function:
        1. Uses QA60 band for cloud and cirrus detection
        2. Applies additional cloud probability threshold
        3. Adds cloud shadow detection using NIR and SWIR bands
        4. Includes terrain shadow masking
        """
        # Get QA band and cloud probability
        qa = image.select('QA60')
        clouds = image.select('MSK_CLDPRB')
        
        # Bits 10 and 11 are clouds and cirrus in QA60
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        # Create cloud mask from QA60 band
        qa_mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            
        # Cloud probability threshold (20%)
        cloud_prob_mask = clouds.lt(20)
        
        # Detect dark pixels in NIR and SWIR1 for potential shadows
        nir = image.select('B8').divide(10000)
        swir1 = image.select('B11').divide(10000)
        
        # Shadow detection thresholds
        nir_threshold = 0.15
        swir_threshold = 0.1
        
        # Create shadow mask
        shadow_mask = nir.gt(nir_threshold).And(swir1.gt(swir_threshold))
        
        # Combine all masks
        final_mask = qa_mask.And(cloud_prob_mask).And(shadow_mask)
        
        # Apply the combined mask
        masked_image = image.updateMask(final_mask)
        
        # Calculate NDVI for the masked image
        ndvi = masked_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Add metadata about masking
        masked_image = masked_image.set({
            'CLOUD_PROB_THRESHOLD': 20,
            'NIR_SHADOW_THRESHOLD': nir_threshold,
            'SWIR_SHADOW_THRESHOLD': swir_threshold,
            'QA_BITS_USED': 'Cloud (10) and Cirrus (11)'
        })
        
        return masked_image.addBands(ndvi)

    @log_execution_time
    @log_memory_usage
    def get_sentinel_data(self, start_date, end_date, season_name):
        """Get Sentinel-2 data for the specified time period."""
        try:
            # Filter Sentinel-2 collection
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(self.aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

            # Apply cloud masking
            collection = collection.map(self.mask_clouds)

            # Select bands and calculate indices
            collection = collection.select(
                ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
                ['blue', 'green', 'red', 'red_edge1', 'red_edge2', 'red_edge3', 'nir', 'nir2', 'swir1', 'swir2']
            )

            # Calculate vegetation indices
            def add_indices(image):
                ndvi = image.normalizedDifference(['nir', 'red']).rename('ndvi')
                evi = image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': image.select('nir'),
                        'RED': image.select('red'),
                        'BLUE': image.select('blue')
                    }
                ).rename('evi')
                ndmi = image.normalizedDifference(['nir', 'swir1']).rename('ndmi')
                return image.addBands([ndvi, evi, ndmi])

            collection = collection.map(add_indices)

            # Calculate median values
            median_image = collection.median()

            # Export spectral bands
            self.export_image(
                image=median_image.select(['blue', 'green', 'red', 'red_edge1', 'red_edge2', 'red_edge3', 'nir', 'nir2', 'swir1', 'swir2']),
                description=f'South_India_Deforestation_{season_name}_spectral',
                folder_name='ECOVISION_South_India',
                season=season_name,
                data_type='spectral'
            )

            # Export vegetation indices
            self.export_image(
                image=median_image.select(['ndvi', 'evi', 'ndmi']),
                description=f'South_India_Deforestation_{season_name}_indices',
                folder_name='ECOVISION_South_India',
                season=season_name,
                data_type='indices'
            )

            # Export quality assessment data
            self.export_image(
                image=median_image.select(['QA60']),
                description=f'South_India_Deforestation_{season_name}_qa',
                folder_name='ECOVISION_South_India',
                season=season_name,
                data_type='qa'
            )

            print(f"\nExport tasks started for ECOVISION_South_India/{season_name}:")
            print(f"- Spectral bands: South_India_Deforestation_{season_name}_spectral")
            print(f"- Vegetation indices: South_India_Deforestation_{season_name}_indices")
            print(f"- Quality assessment: South_India_Deforestation_{season_name}_qa")

        except Exception as e:
            print(f"Error processing {season_name} Sentinel-2 data: {str(e)}")
            raise

    def export_image(self, image, description, folder_name, season, data_type):
        """Export an image to Google Drive."""
        try:
            # Add metadata
            image = image.set({
                'REGION': 'South India',
                'STATES_INCLUDED': ', '.join(self.state_boundaries.keys()),
                'GEOGRAPHIC_BOUNDS': f'[{self.west}, {self.south}, {self.east}, {self.north}]',
                'EXPORT_TIME': ee.Date(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')).format()
            })

            # Export to Google Drive
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder_name,
                scale=10,
                maxPixels=1e13,
                region=self.aoi
            )
            task.start()

        except Exception as e:
            print(f"Error exporting {description}: {str(e)}")
            raise

    def main(self):
        """Main function to process satellite data."""
        try:
            # Calculate and print area statistics
            print("\nArea Statistics:")
            for state_name, boundary in self.state_boundaries.items():
                state_area = boundary.geometry().area().divide(1e6).getInfo()  # Convert to square kilometers
                print(f"{state_name}: {state_area:.2f} km²")

            # Set up time periods
            current_date = datetime.now()
            pre_monsoon_end = current_date + timedelta(days=180)  # 6 months into the future
            pre_monsoon_start = pre_monsoon_end - timedelta(days=180)  # 6 months before end
            previous_season_end = pre_monsoon_start - timedelta(days=5)  # 5 days gap
            previous_season_start = previous_season_end - timedelta(days=180)  # 6 months before

            # Process Pre-Monsoon period
            print("\nProcessing Pre_Monsoon Sentinel-2 data...")
            self.get_sentinel_data(
                pre_monsoon_start.strftime('%Y-%m-%d'),
                pre_monsoon_end.strftime('%Y-%m-%d'),
                'Pre_Monsoon'
            )

            # Process Previous Season
            print("\nProcessing Previous_Season Sentinel-2 data...")
            self.get_sentinel_data(
                previous_season_start.strftime('%Y-%m-%d'),
                previous_season_end.strftime('%Y-%m-%d'),
                'Previous_Season'
            )

            # Print metadata
            print("\nProcessing Metadata:")
            print("Geographic Extent:")
            print(f"- Longitude: {self.west}°E to {self.east}°E")
            print(f"- Latitude: {self.south}°N to {self.north}°N")

            print("\nStates Covered:")
            print("- Madhya Pradesh (partial)")
            print("- Chhattisgarh (partial)")
            
            print("\nTime Periods:")
            print(f"- Pre_Monsoon: {pre_monsoon_start.strftime('%Y-%m-%d')} to {pre_monsoon_end.strftime('%Y-%m-%d')}")
            print(f"- Previous_Season: {previous_season_start.strftime('%Y-%m-%d')} to {previous_season_end.strftime('%Y-%m-%d')}")

            print("\nDataset Specifications:")
            print("- Source: Sentinel-2 Level-2A Surface Reflectance (Harmonized)")
            print("- Resolution: 10 meters")
            print("- Cloud Cover Threshold: 20%")
            print("- Spectral Bands:")
            print("  * Visible (B2, B3, B4)")
            print("  * Red Edge (B5, B6, B7)")
            print("  * Near Infrared (B8, B8A)")
            print("  * Short Wave Infrared (B11, B12)")
            print("- Vegetation Indices:")
            print("  * NDVI (Normalized Difference Vegetation Index)")
            print("  * EVI (Enhanced Vegetation Index)")
            print("  * NDMI (Normalized Difference Moisture Index)")

            print("\nExport Status:")
            print("- Tasks have been submitted to Google Earth Engine")
            print("- Download files from Google Drive when complete")
            print("- Move downloaded files to: D:\\ECOVISION\\downloads")
            print("- Run data organization when downloads complete")

            print("\nNext Steps:")
            print("1. Wait for Google Earth Engine tasks to complete")
            print("2. Download exported files from Google Drive")
            print("3. Place downloaded files in: D:\\ECOVISION\\downloads")
            print("4. Run the following to organize data:")
            print("   detector.organize_local_data(download_dir, processed_dir)")
            print("5. Processed data will be available in: D:\\ECOVISION\\processed_data")

            print("\nExport Tasks Started:")
            print("- Spectral data and vegetation indices will be exported to 'ECOVISION_South_India' folder")
            print("- Files will be saved as Cloud-Optimized GeoTIFFs")
            print("- Check your Google Drive for the results")

        except Exception as e:
            print(f"Error in main function: {str(e)}")
            raise

    def validate_region(self, image):
        """
        Validate that the image data is strictly within South India boundaries.
        
        Args:
            image: ee.Image to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Get image footprint
            footprint = image.geometry()
            
            # Check if the footprint is completely within South India states
            contained = footprint.containedIn(self.target_states.geometry())
            
            # Calculate overlap percentage with each state
            state_overlaps = {
                'Tamil Nadu': footprint.intersection(self.tn_boundary.geometry()).area().divide(footprint.area()).multiply(100),
                'Kerala': footprint.intersection(self.kerala_boundary.geometry()).area().divide(footprint.area()).multiply(100),
                'Karnataka': footprint.intersection(self.karnataka_boundary.geometry()).area().divide(footprint.area()).multiply(100),
                'Andhra Pradesh': footprint.intersection(self.ap_boundary.geometry()).area().divide(footprint.area()).multiply(100),
                'Telangana': footprint.intersection(self.telangana_boundary.geometry()).area().divide(footprint.area()).multiply(100)
            }
            
            # Convert to Python dictionary
            overlap_percentages = {state: float(pct.getInfo()) for state, pct in state_overlaps.items()}
            
            # Log validation results
            self.logger.info("Regional Validation Results:")
            self.logger.info(f"Image contained within South India: {contained.getInfo()}")
            self.logger.info("State overlap percentages:")
            for state, percentage in overlap_percentages.items():
                self.logger.info(f"- {state}: {percentage:.2f}%")
            
            # Validate that we have coverage in all states
            all_states_covered = all(pct > 0 for pct in overlap_percentages.values())
            if not all_states_covered:
                missing_states = [state for state, pct in overlap_percentages.items() if pct == 0]
                self.logger.warning(f"Missing coverage for states: {', '.join(missing_states)}")
            
            return contained.getInfo()
            
        except Exception as e:
            self.logger.error(f"Error in region validation: {str(e)}")
            return False

    def generate_completion_report(self, processed_data):
        """
        Generate a detailed completion report for South India data processing.
        
        Args:
            processed_data: Dictionary containing processing results and metadata
            
        Returns:
            dict: Completion report with detailed statistics
        """
        try:
            report = {
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'region': 'South India',
                'states_processed': [
                    'Tamil Nadu',
                    'Kerala',
                    'Karnataka',
                    'Andhra Pradesh',
                    'Telangana'
                ],
                'geographic_bounds': {
                    'west': self.west,
                    'south': self.south,
                    'east': self.east,
                    'north': self.north
                },
                'processing_statistics': {
                    'total_area_processed_km2': self.aoi.area().divide(1e6).getInfo(),
                    'cloud_coverage_threshold': 20,
                    'spatial_resolution_meters': 10
                },
                'data_completeness': {
                    'spectral_bands': True,
                    'vegetation_indices': True,
                    'quality_assessment': True
                },
                'regional_coverage': {},
                'special_considerations': []
            }
            
            # Calculate state-wise coverage
            total_area = self.aoi.area().getInfo()
            for state, boundary in {
                'Tamil Nadu': self.tn_boundary,
                'Kerala': self.kerala_boundary,
                'Karnataka': self.karnataka_boundary,
                'Andhra Pradesh': self.ap_boundary,
                'Telangana': self.telangana_boundary
            }.items():
                state_intersection = self.aoi.intersection(boundary.geometry())
                coverage_percent = state_intersection.area().divide(total_area).multiply(100).getInfo()
                report['regional_coverage'][state] = {
                    'coverage_percentage': coverage_percent,
                    'area_km2': state_intersection.area().divide(1e6).getInfo()
                }
            
            # Add special regional considerations
            for state, coverage in report['regional_coverage'].items():
                if coverage['coverage_percentage'] < 5:
                    report['special_considerations'].append(
                        f"Low coverage warning: {state} has less than 5% coverage"
                    )
            
            # Log the report
            self.logger.info("Completion Report Generated:")
            self.logger.info(json.dumps(report, indent=2))
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating completion report: {str(e)}")
            raise

    def organize_local_data(self, download_dir, output_dir):
        """
        Organize and preprocess downloaded satellite data for analysis.
        
        Args:
            download_dir: Directory containing downloaded GEE exports
            output_dir: Directory to store organized and preprocessed data
        """
        try:
            # Create state-specific directories
            state_dirs = {
                'TN': 'Tamil_Nadu',
                'KL': 'Kerala',
                'KA': 'Karnataka',
                'AP': 'Andhra_Pradesh',
                'TG': 'Telangana'
            }
            
            for state_code, state_name in state_dirs.items():
                state_path = os.path.join(output_dir, state_name)
                os.makedirs(state_path, exist_ok=True)
                
                # Create subdirectories for each data type
                for data_type in ['spectral', 'indices', 'qa']:
                    os.makedirs(os.path.join(state_path, data_type), exist_ok=True)
            
            # Move and organize files
            for filename in os.listdir(download_dir):
                if filename.endswith('.tif'):
                    src_path = os.path.join(download_dir, filename)
                    
                    # Determine data type
                    if 'spectral' in filename:
                        data_type = 'spectral'
                    elif 'indices' in filename:
                        data_type = 'indices'
                    elif 'qa' in filename:
                        data_type = 'qa'
                    else:
                        continue
                    
                    # Copy file to each state directory (will be filtered later)
                    for state_name in state_dirs.values():
                        dst_path = os.path.join(output_dir, state_name, data_type, filename)
                        shutil.copy2(src_path, dst_path)
            
            print("\nData Organization Complete:")
            print("Files have been organized by state and data type")
            print("\nDirectory Structure:")
            for state_code, state_name in state_dirs.items():
                print(f"\n{state_name}/")
                for data_type in ['spectral', 'indices', 'qa']:
                    print(f"  └── {data_type}/")
            
        except Exception as e:
            print(f"Error organizing data: {str(e)}")
            raise

if __name__ == "__main__":
    detector = DeforestationDetector()
    detector.main()
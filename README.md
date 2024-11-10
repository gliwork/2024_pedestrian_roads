# 2024_pedestrian_roads
2024 International hackaton


docker build -t geoprocessing-app .
docker run --rm -v /path/to/your/data/vector:/app/vector geoprocessing-app \
    --vector_path vector/ \
    --output_csv speed_array_excel_file.csv \
    --sample_matrix

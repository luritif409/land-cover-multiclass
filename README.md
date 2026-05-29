https://www.kaggle.com/datasets/apollo2506/eurosat-dataset?resource=download


1. running dengan command :
cd D:\LAB_RESEARCH\code\land-cover-multiclass-classification

.\Scripts\python.exe scripts\run_plan.py --plan configs\pc1_resnet50_remaining.csv --rgb-splits-dir outputs\splits_rgb --multispectral-splits-dir outputs\splits_allbands --rgb-dataset-dir D:\LAB_RESEARCH\code\land-cover-multiclass-classification\src\datasets\EuroSATrgb --multispectral-dataset-dir D:\LAB_RESEARCH\code\land-cover-multiclass-classification\src\datasets\EuroSATallBands --no-pretrained

2. update file data.py di direktori
   E:\LAB_RESEARCH\code\land-cover-multiclass-classification\src\eurosat_research

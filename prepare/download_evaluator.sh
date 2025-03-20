mkdir -p ./checkpoints/eval_model/
cd ./checkpoints/eval_model/
echo "The pretrained evaluation model will be stored in the './checkpoints/eval_model/' folder\n"
# InterHuman
echo "Downloading the evaluation model..."
gdown https://drive.google.com/uc?id=1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH
gdown https://drive.google.com/uc?id=13NzdH-9xSOhrOCEVY8lmYrSzebZNv5xz
echo "Evaluation model downloaded"
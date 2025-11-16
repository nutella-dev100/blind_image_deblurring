from model import DeblurModel
import config
import os

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    
    print("starting blind deblurring")
    model = DeblurModel(config)
    result = model.run(
        image_path="test_images/test.png",
        save_path=config.RESULT_PATH
    )
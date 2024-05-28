from model import Segmentator
import my_custom_dataset

model = Segmentator()

# Defining datasets
train_dataset = model.load_dataset('train')
val_dataset = model.load_dataset('val')

# Initialize our model
model.build_model()

# Training (built-in method fit )
model.train_model(train_dataset, val_dataset)

# Saving the model
model.save_model('trained_model_t800.keras')

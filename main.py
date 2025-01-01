import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import json
import numpy as np

class BodyPartDataset(Dataset):
    """Dataset for medical body part images"""
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('L')
        except Exception as e:
            print(f"Failed to load image: {str(e)}")
            image = Image.new('L', (224, 224), 0)  # Return blank image if loading fails
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]



class BodyPartIdentificationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Medical Image Identification")
        self.master.geometry("800x800")

        # Define the specific body parts we want to identify
        self.body_parts = ['brain', 'lungs', 'knee', 'hand']
        self.part_to_idx = {part: idx for idx, part in enumerate(self.body_parts)}
        
        # Training data storage
        self.training_data = {part: [] for part in self.body_parts}
        
        # Initialize training metrics
        self.training_accuracy = []
        
        # Model file path
        self.model_path = 'medicallll_model.pth'
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transforms
        self.transform = self.get_preprocessing()
        self.train_transform = self.get_training_transforms()
        
        # Initialize or load model
        self.initialize_model()
        
        self.create_widgets()
        self.load_training_data()

    def get_preprocessing(self):
        """Returns the preprocessing pipeline for inference"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.456], std=[0.224]),
            transforms.Lambda(lambda x: torch.clamp(x * 1.2, 0, 1))
        ])

    def get_training_transforms(self):
        """Returns the preprocessing pipeline with augmentations for training"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.456], std=[0.224]),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3)
        ])

    def initialize_model(self):
        self.model = models.resnet18(pretrained=True)
        
        # Modify first layer for single-channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                   stride=(2, 2), padding=(3, 3), bias=False)
        
        # Enhanced final classifier
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), #allowing it to learn complex relationships in the data
            nn.Dropout(0.3), #This helps prevent overfitting by reducing reliance on specific neurons
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.body_parts))
        )
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                print("Loaded existing model")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.initialize_new_model()
        else:
            self.initialize_new_model()
        
        self.model = self.model.to(self.device)

    def initialize_new_model(self):
        """Initialize a new model with frozen layers except final layers"""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the final layers
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Style configuration
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        
        # Create frames
        self.top_frame = ttk.Frame(self.master)
        self.top_frame.pack(pady=10, padx=10, fill='x')
        
        self.middle_frame = ttk.Frame(self.master)
        self.middle_frame.pack(pady=10, padx=10, fill='x')
        
        self.display_frame = ttk.Frame(self.master)
        self.display_frame.pack(pady=10, padx=10, expand=True, fill='both')

        # Header
        header_label = ttk.Label(self.top_frame, 
                               text="Medical Image Identification System", 
                               style='Header.TLabel')
        header_label.pack(pady=10)

        # Training Controls Frame
        training_frame = ttk.LabelFrame(self.middle_frame, text="Training Controls")
        training_frame.pack(fill='x', padx=5, pady=5)

        # Body part selection dropdown
        self.selected_part = tk.StringVar()
        ttk.Label(training_frame, text="Select Body Part:").pack(side='left', padx=5)
        self.part_dropdown = ttk.Combobox(training_frame, 
                                        textvariable=self.selected_part,
                                        values=self.body_parts,
                                        width=15)
        self.part_dropdown.pack(side='left', padx=5)

        # Training control buttons
        self.add_training_button = ttk.Button(training_frame, 
                                            text="Add Training Images", 
                                            command=self.add_training_images)
        self.add_training_button.pack(side='left', padx=5)

        self.train_button = ttk.Button(training_frame, 
                                     text="Train Model", 
                                     command=self.train_model)
        self.train_button.pack(side='left', padx=5)

        # Testing Controls Frame
        testing_frame = ttk.LabelFrame(self.middle_frame, text="Testing Controls")
        testing_frame.pack(fill='x', padx=5, pady=5)

        self.test_button = ttk.Button(testing_frame, 
                                    text="Test New Image", 
                                    command=self.import_image)
        self.test_button.pack(side='left', padx=5)

        # Status and accuracy labels
        self.status_label = ttk.Label(testing_frame, text="Status: Ready")
        self.status_label.pack(side='left', padx=5)
        
        self.accuracy_label = ttk.Label(testing_frame, text="Model Accuracy: N/A")
        self.accuracy_label.pack(side='left', padx=5)
        
        # Image display
        self.image_frame = ttk.LabelFrame(self.display_frame, text="Image Preview")
        self.image_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=10)

        # Result label
        self.result_label = ttk.Label(self.display_frame, 
                                    text="", 
                                    font=('Helvetica', 12))
        self.result_label.pack(pady=10)

    def train_model(self):
        """Train the model using the collected training data"""
        total_images = sum(len(images) for images in self.training_data.values())
        if total_images == 0:
            messagebox.showerror("Error", "No training data available!")
            return

        self.training_accuracy = []
        all_images = []
        all_labels = []
        
        for part, image_paths in self.training_data.items():
            all_images.extend(image_paths)
            all_labels.extend([self.part_to_idx[part]] * len(image_paths))

        dataset = BodyPartDataset(
            image_paths=all_images,
            labels=all_labels,
            transform=self.train_transform,
            is_training=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 10
        try:
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0

                for i, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if i % 2 == 0:
                        self.status_label.config(
                            text=f"Epoch {epoch+1}/{num_epochs}, "
                            f"Batch {i+1}/{len(dataloader)}, "
                            f"Loss: {running_loss/(i+1):.3f}"
                        )
                        self.master.update()

                epoch_accuracy = 100 * correct / total
                self.training_accuracy.append(epoch_accuracy)
                
                avg_accuracy = sum(self.training_accuracy[-5:]) / min(len(self.training_accuracy), 5)
                self.accuracy_label.config(text=f"Model Accuracy: {avg_accuracy:.1f}%")
                
                self.status_label.config(
                    text=f"Completed Epoch {epoch+1}/{num_epochs}, "
                    f"Accuracy: {epoch_accuracy:.1f}%"
                )
                self.master.update()

            self.save_model()
            
            self.status_label.config(text="Training completed and model saved!")
            messagebox.showinfo("Training Complete", 
                              f"Model trained for {num_epochs} epochs\n"
                              f"Final Accuracy: {epoch_accuracy:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")
            self.status_label.config(text="Training failed!")

    def save_model(self):
        """Save the model to disk"""
        torch.save(self.model.state_dict(), self.model_path)

    def add_training_images(self):
        """Add new training images for the selected body part"""
        if not self.selected_part.get():
            messagebox.showerror("Error", "Please select a body part first!")
            return

        files = filedialog.askopenfilenames(
            title=f"Select {self.selected_part.get()} images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if files:
            self.training_data[self.selected_part.get()].extend(files)
            self.save_training_data()
            self.status_label.config(
                text=f"Added {len(files)} images for {self.selected_part.get()}"
            )

    def save_training_data(self):
        """Save training data paths to JSON file"""
        data = {part: list(paths) for part, paths in self.training_data.items()}
        with open('training_data.json', 'w') as f:
            json.dump(data, f)

    def load_training_data(self):
        """Load training data paths from JSON file"""
        try:
            with open('training_data.json', 'r') as f:
                loaded_data = json.load(f)
                self.training_data = {part: loaded_data.get(part, []) 
                                    for part in self.body_parts}
        except FileNotFoundError:
            self.training_data = {part: [] for part in self.body_parts}

    def import_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image to identify",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # Display the image
            image = Image.open(file_path)
            # Set fixed size for display
            display_size = (400, 400)
            image = image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference!

            # Predict
            self.identify_part(file_path)

    def identify_part(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        
        # Show prediction with percentage
        predicted_probability = probabilities[predicted_class].item() * 100
        result = f"Prediction: {self.body_parts[predicted_class].upper()} ({predicted_probability:.2f}%)"
        self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = BodyPartIdentificationApp(root)
    root.mainloop()
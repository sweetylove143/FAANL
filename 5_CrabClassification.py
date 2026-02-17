import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load CSV
df = pd.read_csv('./crabs.csv')
  
# Drop unnecessary columns  
df = df.drop(columns=['rownames', 'index'])  
  
# Encode categorical columns  
label_encoders = {}  
for column in ['sp', 'sex']:  
    label_encoders[column] = LabelEncoder()  
    df[column] = label_encoders[column].fit_transform(df[column])  
  
# Features and target  
X = df.drop(columns=['sp']).values  
y = df['sp'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define model (same architecture)
class CrabNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 species

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax (handled by loss)
        return x

model = CrabNet(X_train.shape[1])

# Equivalent to:
# optimizer='adam'
# loss='sparse_categorical_crossentropy'
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training (equivalent to model.fit)
epochs = 50
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation (equivalent to model.evaluate)
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f"Test Accuracy: {accuracy:.4f}")

# 1. Install and load required libraries
install.packages(c("e1071", "caret", "dplyr", "ggplot2", "pROC", "VIM", "ROSE", "randomForest"))
library(e1071)
library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(VIM)
library(ROSE)
library(randomForest)

# 2. Load dataset
stroke_data <- read.csv("C:/Users/R E V O/Downloads/stroke_prediction_dataset.csv")

# 3. Explore dataset
str(stroke_data)
summary(stroke_data)

# 4. Identify missing values 
cat("Total missing values:", sum(is.na(stroke_data)), "\n")

# 5. Remove rows with missing values
stroke_data <- na.omit(stroke_data)

# 6. Convert categorical columns to factors
categorical_cols <- c("Gender", "Marital.Status", "Work.Type", "Residence.Type", 
                      "Smoking.Status", "Alcohol.Intake", "Physical.Activity", 
                      "Stroke.History", "Family.History.of.Stroke", "Dietary.Habits", 
                      "Diagnosis")  # Removed problematic columns

stroke_data[categorical_cols] <- lapply(stroke_data[categorical_cols], as.factor)

# 7. Extract numeric values from Blood Pressure Levels
stroke_data$Systolic_BP <- as.numeric(sub("([0-9]+)/([0-9]+)", "\\1", stroke_data$Blood.Pressure.Levels))
stroke_data$Diastolic_BP <- as.numeric(sub("([0-9]+)/([0-9]+)", "\\2", stroke_data$Blood.Pressure.Levels))
stroke_data <- stroke_data %>% select(-Blood.Pressure.Levels)

# 8. Extract numeric values from Cholesterol Levels
stroke_data$LDL <- as.numeric(sub(".*LDL: ([0-9]+).*", "\\1", stroke_data$Cholesterol.Levels))
stroke_data$HDL <- as.numeric(sub(".*HDL: ([0-9]+).*", "\\1", stroke_data$Cholesterol.Levels))
stroke_data <- stroke_data %>% select(-Cholesterol.Levels)

# 9. Drop unnecessary columns
stroke_data <- stroke_data %>% select(-Patient.ID, -Patient.Name, -Symptoms)

# 10. Visualize class distribution
ggplot(stroke_data, aes(x = Diagnosis, fill = Diagnosis)) +
  geom_bar() +
  labs(title = "Distribution of Stroke Cases", x = "Diagnosis", y = "Count") +
  scale_fill_manual(values = c("red", "green")) +
  theme_classic()

# 11. Train-test split (80/20)
set.seed(123)
splitIndex <- createDataPartition(stroke_data$Diagnosis, p = 0.8, list = FALSE)
train_data <- stroke_data[splitIndex, ]
test_data <- stroke_data[-splitIndex, ]

# 12. Balance training data using ROSE
train_data_balanced <- ovun.sample(Diagnosis ~ ., data = train_data, method = "both", p = 0.5, seed = 123)$data

# 13. Standardize numeric features
numeric_cols <- sapply(train_data_balanced, is.numeric)
preProcValues <- preProcess(train_data_balanced[, numeric_cols], method = c("center", "scale"))
train_data_balanced[, numeric_cols] <- predict(preProcValues, train_data_balanced[, numeric_cols])
test_data[, numeric_cols] <- predict(preProcValues, test_data[, numeric_cols])

# 14. Check categorical variables before training
cat("Unique categories per factor:\n")
sapply(train_data_balanced[, sapply(train_data_balanced, is.factor)], function(x) length(unique(x)))

# ------------------------
# RANDOM FOREST MODEL
# ------------------------

# 15. Train Random Forest model
set.seed(123)
rf_model <- randomForest(Diagnosis ~ ., data = train_data_balanced, ntree = 100, importance = TRUE)

# 16. Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# 17. Compute the confusion matrix
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Diagnosis)
print(rf_conf_matrix)

# 18. Visualize Random Forest confusion matrix
rf_conf <- as.data.frame(rf_conf_matrix$table)
ggplot(rf_conf, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(title = "Random Forest Confusion Matrix", x = "True Labels", y = "Predicted Labels") +
  theme_minimal()

# 19. ROC Curve for Random Forest
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")
rf_roc <- roc(response = test_data$Diagnosis, predictor = rf_probs[, 2])
plot(rf_roc, main = "Random Forest ROC Curve", col = "darkgreen", lwd = 2)
auc_rf <- auc(rf_roc)
cat("Random Forest AUC:", auc_rf, "\n")

# 20. Extract and print accuracy
rf_accuracy <- rf_conf_matrix$overall['Accuracy']
rf_accuracy_percent <- round(rf_accuracy * 100, 2)
cat("Random Forest Accuracy:", rf_accuracy_percent, "%\n")


# 1. Install and load required libraries
install.packages(c("e1071", "caret", "dplyr", "ggplot2", "pROC", "VIM", "ROSE", "naniar"))

# 2. Load libraries

library(e1071)     # For Support Vector Machines (SVM) and other machine learning models
library(caret)     # For training and evaluating machine learning models (classification and regression)
library(dplyr)     # For data manipulation (filtering, selecting, mutating, summarizing data frames)
library(ggplot2)   # For data visualization (creating elegant and informative plots)
library(pROC)      # For ROC curve analysis and calculating AUC (Area Under the Curve)
library(VIM)       # For visualizing and imputing missing values in datasets
library(ROSE)      # For dealing with imbalanced datasets (oversampling, undersampling, synthetic data generation)
library(naniar)    # For easy visualization and handling of missing data

# 3. Load dataset
stroke_data <- read.csv("C:/Users/R E V O/Downloads/stroke_prediction_dataset.csv")

# 4. Explore dataset C:/Users/R E V O/Documents/stroke-prediction/script1.R
str(stroke_data)
summary(stroke_data)

# 5. Visualize missing data
gg_miss_var(stroke_data) +
  labs(title = "Missing Values per Column",
       x = "Variables",
       y = "Number of Missing Values") +
  theme_minimal()

aggr(stroke_data, col = c("navy", "yellow"), numbers = TRUE,
     prop = c(TRUE, FALSE), labels = names(stroke_data),
     cex.axis = 0.7, cex.numbers = 0.8, gap = 3,
     ylab = c("Missing data", "Patterns"))

cat("Missing Values After Handling:\n")
print(colSums(is.na(stroke_data)))

# 6. Remove rows with missing values (or use imputation)
stroke_data <- na.omit(stroke_data)

# 7. Convert categorical columns to factors
categorical_cols <- c("Gender", "Marital.Status", "Work.Type", "Residence.Type", 
                      "Smoking.Status", "Alcohol.Intake", "Physical.Activity", 
                      "Stroke.History", "Family.History.of.Stroke", "Dietary.Habits", 
                      "Stress.Levels", "Blood.Pressure.Levels", "Cholesterol.Levels", 
                      "Symptoms", "Diagnosis")
stroke_data[categorical_cols] <- lapply(stroke_data[categorical_cols], as.factor)

# 8. Drop unnecessary columns
stroke_data <- stroke_data %>% select(-Patient.ID, -Patient.Name)

# 9. Visualize class distribution
ggplot(stroke_data, aes(x = Diagnosis, fill = Diagnosis)) +
  geom_bar() +
  labs(title = "Distribution of Stroke Cases", x = "Diagnosis", y = "Count") +
  scale_fill_manual(values = c("No Stroke" = "red", "Stroke" = "green")) +
  theme_classic()

# 10. Train-test split (80/20)
set.seed(123)
splitIndex <- createDataPartition(stroke_data$Diagnosis, p = 0.8, list = FALSE)
train_data <- stroke_data[splitIndex, ]
test_data <- stroke_data[-splitIndex, ]

# 11. Balance training data using ROSE
train_data_balanced <- ovun.sample(Diagnosis ~ ., data = train_data, method = "both", p = 0.5, seed = 123)$data

# 12. Standardize numeric features
numeric_cols <- sapply(train_data_balanced, is.numeric)
preProcValues <- preProcess(train_data_balanced[, numeric_cols], method = c("center", "scale"))
train_data_balanced[, numeric_cols] <- predict(preProcValues, train_data_balanced[, numeric_cols])
test_data[, numeric_cols] <- predict(preProcValues, test_data[, numeric_cols])

# 13. Train SVM model
svm_model <- svm(Diagnosis ~ ., data = train_data_balanced, kernel = "radial", cost = 1, scale = FALSE)
print(svm_model)

# 14. Predict on test set
svm_predictions <- predict(svm_model, newdata = test_data)

# 15. Evaluate SVM using confusion matrix
conf_matrix <- confusionMatrix(svm_predictions, test_data$Diagnosis)
print(conf_matrix)

# 16. Visualize SVM confusion matrix
confusion <- as.data.frame(conf_matrix$table)
ggplot(confusion, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "SVM Confusion Matrix", x = "True Labels", y = "Predicted Labels") +
  theme_minimal()

# 17. ROC Curve for SVM
svm_roc <- roc(response = test_data$Diagnosis, predictor = as.numeric(svm_predictions))
plot(svm_roc, main = "SVM ROC Curve", col = "blue", lwd = 2)
auc(svm_roc)

# 18. Logistic Regression Model (Comparison)
logit_model <- glm(Diagnosis ~ ., data = train_data_balanced, family = "binomial",
                   control = glm.control(maxit = 100))
summary(logit_model)

# 19. Predict with Logistic Regression
logit_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_pred_classes <- ifelse(logit_probs > 0.5, "Stroke", "No Stroke")
logit_pred_classes <- factor(logit_pred_classes, levels = levels(test_data$Diagnosis))

# 20. Confusion Matrix for Logistic Regression
logit_conf_matrix <- confusionMatrix(logit_pred_classes, test_data$Diagnosis)
print(logit_conf_matrix)

# 21. ROC Curve for Logistic Regression
logit_roc <- roc(response = test_data$Diagnosis, predictor = logit_probs)
plot(logit_roc, main = "Logistic Regression ROC Curve", col = "darkgreen", lwd = 2)
auc(logit_roc)

# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(VIM)
library(ROSE)
library(naniar)
library(corrplot)

# Load dataset
stroke_data <- read.csv("C:/Users/R E V O/Downloads/stroke_prediction_dataset.csv")

# Explore dataset
str(stroke_data)
summary(stroke_data)

# Visualize missing data
aggr(stroke_data,
     numbers = TRUE,
     sortVars = TRUE,
     cex.axis = 0.7,
     gap = 3,
     ylab = c("Missing Data", "Pattern"))

# Check missing values count
data.frame(
  Column = colnames(stroke_data),
  Missing_Count = colSums(is.na(stroke_data))
)

# Convert categorical columns to factors
categorical_cols <- c("Gender", "Marital.Status", "Work.Type", "Residence.Type", 
                      "Smoking.Status", "Alcohol.Intake", "Physical.Activity", 
                      "Stroke.History", "Family.History.of.Stroke", "Dietary.Habits", 
                      "Stress.Levels", "Blood.Pressure.Levels", "Cholesterol.Levels", 
                      "Symptoms", "Diagnosis")
stroke_data[categorical_cols] <- lapply(stroke_data[categorical_cols], as.factor)

# Drop unnecessary columns
stroke_data <- stroke_data %>% select(-Patient.ID, -Patient.Name)

# ===================
# Binning begins here
# ===================

# Convert Stress.Levels to numeric and bin
stroke_data$Stress.Levels <- as.numeric(as.character(stroke_data$Stress.Levels))
stroke_data$Stress.Category <- cut(stroke_data$Stress.Levels,
                                   breaks = c(-Inf, 3.3, 6.6, Inf),
                                   labels = c("Low", "Medium", "High"))

# Extract LDL and HDL values from the Cholesterol.Levels column
stroke_data$LDL <- as.numeric(sub(".*LDL: ([0-9]+).*", "\\1", stroke_data$Cholesterol.Levels))
stroke_data$HDL <- as.numeric(sub(".*HDL: ([0-9]+).*", "\\1", stroke_data$Cholesterol.Levels))

# Bin LDL values based on clinical guidelines
stroke_data$LDL.Category <- cut(stroke_data$LDL,
                                breaks = c(-Inf, 100, 129, 159, 189, Inf),
                                labels = c("Optimal", "Near Optimal", "Borderline High", "High", "Very High"))

# Bin HDL values based on clinical guidelines
stroke_data$HDL.Category <- cut(stroke_data$HDL,
                                breaks = c(-Inf, 39, 60, Inf),
                                labels = c("Low", "Normal", "High"))

# Combine LDL and HDL bins into one factor
stroke_data$Cholesterol.Levels <- paste("LDL:", stroke_data$LDL.Category, "HDL:", stroke_data$HDL.Category)
stroke_data$Cholesterol.Levels <- as.factor(stroke_data$Cholesterol.Levels)

# Remove raw LDL and HDL columns if not needed
stroke_data <- stroke_data %>% select(-LDL, -HDL)

head(unique(stroke_data$Blood.Pressure.Levels), 20)


# ===================
# Split data
# ===================

set.seed(123)
splitIndex <- createDataPartition(stroke_data$Diagnosis, p = 0.8, list = FALSE)
train_data <- stroke_data[splitIndex, ]
test_data <- stroke_data[-splitIndex, ]

# Balance the training data using ROSE
train_data_balanced <- ovun.sample(Diagnosis ~ ., data = train_data, method = "both", p = 0.5, seed = 123)$data

# Convert numeric columns to numeric if stored as characters
numeric_cols <- sapply(train_data_balanced, is.numeric)
train_data_balanced[, numeric_cols] <- lapply(train_data_balanced[, numeric_cols], as.numeric)
test_data[, numeric_cols] <- lapply(test_data[, numeric_cols], as.numeric)

# Standardize numeric features
preProcValues <- preProcess(train_data_balanced[, numeric_cols], method = c("center", "scale"))
train_data_balanced[, numeric_cols] <- predict(preProcValues, train_data_balanced[, numeric_cols])
test_data[, numeric_cols] <- predict(preProcValues, test_data[, numeric_cols])

# Check categorical predictors with too many levels
sapply(train_data_balanced, function(x) if(is.factor(x)) length(levels(x)) else NA)



# ===================
# Model training
# ===================

set.seed(123)
rf_model <- randomForest(Diagnosis ~ ., data = train_data_balanced, ntree = 100, importance = TRUE)
print(rf_model)

# Predict on test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Confusion matrix and accuracy
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Diagnosis)
print(rf_conf_matrix)
cat("✅ Random Forest Accuracy:", round(rf_conf_matrix$overall['Accuracy'] * 100, 2), "%\n")

# Plot ROC curve
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")
rf_roc <- roc(response = test_data$Diagnosis, predictor = rf_probs[, "Stroke"])
plot(rf_roc, main = "Random Forest ROC Curve", col = "purple", lwd = 2)
auc(rf_roc)

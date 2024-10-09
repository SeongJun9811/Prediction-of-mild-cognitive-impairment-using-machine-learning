install.packages('foreign')
install.packages("caret")
install.packages("pROC")
install.packages("randomForest")
install.packages("nnet")
install.packages("e1071")
install.packages("shapr")
library(shapr)
library(caret)
library(pROC)
library(foreign)
library(randomForest)
library(nnet)
library(e1071)
library(dplyr)
library(tidyr)

dat <- read.spss('C:/Users/seongjun/Downloads/consult.sav', to.data.frame=T, reencode='utf-8')
#View(dat)
head(dat)

################################################################################
# 데이터 전처리
set.seed(123)

# 훈련용 검증요 데이터 분리
train_index <- createDataPartition(dat$MMSE, p = 0.7, list = FALSE)
train_data <- dat[train_index, ]
test_data  <- dat[-train_index, ]

# 범주형 변수 변환
train_data$MMSE <- as.factor(train_data$MMSE)
train_data$edu <- as.factor(train_data$edu)
test_data$MMSE <- as.factor(test_data$MMSE)
test_data$edu <- as.factor(test_data$edu)

str(train_data)
################################################################################
# 1. 로지스틱 회귀분석
set.seed(123)
# train data를 이용해 로지스틱 회귀모형 적합
logit_model <- glm(MMSE ~ gender + age + edu + marital + religion + como + time + BMI + interaction +
                     health + restict + smoking + alco + meal + PA + CESD + ADL + IADL, 
                   data = train_data, family = binomial)

# 로지스틱 회귀모형 결과
summary(logit_model)

# 훈련용 데이터 예측
predictions <- predict(logit_model, newdata = train_data, type = "response")
predicted_class <- ifelse(predictions > 0.5, "MCI", "N-S")
confusion_matrix <- table(Predicted = predicted_class, Actual = train_data$MMSE)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("분류정확도:", 1 - accuracy))


# 검증용 데이터 예측
predictions <- predict(logit_model, newdata = test_data, type = "response")
predicted_class <- ifelse(predictions > 0.5, "MCI", "N-S")
confusion_matrix <- table(Predicted = predicted_class, Actual = test_data$MMSE)

# 혼동행렬 생성
print(confusion_matrix)

# 분류정확도 계산
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("분류정확도:", 1 - accuracy))

# ROC CURVE
roc_curve <- roc(test_data$MMSE, predictions, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix["MCI", "MCI"]
TN <- confusion_matrix["N-S", "N-S"]
FP <- confusion_matrix["MCI", "N-S"]
FN <- confusion_matrix["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

##########
# 로지스틱 회귀 중요 변수 선정 후 모델 적합
# 유의미한 변수 : age, edu, time, interaction, smoking, CESD
set.seed(123)
logit_model2 <- glm(MMSE ~ age + edu + time + interaction +
                     smoking + CESD , 
                   data = train_data, family = binomial)

# 훈련용 데이터 예측
predictions <- predict(logit_model2, newdata = train_data, type = "response")
predicted_class <- ifelse(predictions > 0.5, "MCI", "N-S")
confusion_matrix <- table(Predicted = predicted_class, Actual = train_data$MMSE)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("분류정확도:", 1 - accuracy))


# 검증용 데이터 예측
predictions <- predict(logit_model2, newdata = test_data, type = "response")
predicted_class <- ifelse(predictions > 0.5, "MCI", "N-S")
confusion_matrix <- table(Predicted = predicted_class, Actual = test_data$MMSE)
print(confusion_matrix)

# 분류정확도 계산
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("분류정확도:", 1 - accuracy))

# ROC CURVE
roc_curve <- roc(test_data$MMSE, predictions, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix["MCI", "MCI"]
TN <- confusion_matrix["N-S", "N-S"]
FP <- confusion_matrix["MCI", "N-S"]
FN <- confusion_matrix["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

################################################################################
# 2. 랜포
set.seed(123)

# 모델 적합
rf_model <- randomForest(MMSE ~ gender + age + edu + marital + religion + como + time + BMI + interaction +
                           health + restict + smoking + alco + meal + PA + CESD + ADL + IADL, 
                         data = train_data, ntree = 500, importance = TRUE)

# 테스트 데이터 예측
rf_model$confusion
accuracy_rf <- sum(diag(rf_model$confusion)) / sum(rf_model$confusion)
print(paste("Accuracy: ", accuracy_rf))

# 변수중요도 확인
importance(rf_model)
varImpPlot(rf_model)

# 랜덤포레스트 활용 예측
rf_predictions <- predict(rf_model, newdata = test_data)
confusion_matrix_rf <- table(Predicted = rf_predictions, Actual = test_data$MMSE)
confusion_matrix_rf

rf_probabilities <- predict(rf_model, newdata = test_data, type = "prob")
rf_predictions_prob <- rf_probabilities[, "MCI"]

# 분류 정확도
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy: ", accuracy_rf))

# roc curve
roc_curve <- roc(test_data$MMSE, rf_predictions_prob, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "Random Forest ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_rf["MCI", "MCI"]
TN <- confusion_matrix_rf["N-S", "N-S"]
FP <- confusion_matrix_rf["MCI", "N-S"]
FN <- confusion_matrix_rf["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

##########
# 랜포 중요 변수 선정 후 모델 적합
# 유의미한 변수 : age, , bmi, edu, time, interaction, CESD, como , meal, IADL, health
set.seed(123)
rf_model2 <- randomForest(MMSE ~ age + edu + como + time + BMI + interaction +
                            CESD + meal + IADL+ health, 
                         data = train_data, ntree = 500, importance = TRUE)

# 테스트 데이터 예측
rf_model2$confusion
accuracy_rf <- sum(diag(rf_model2$confusion)) / sum(rf_model2$confusion)
print(paste("Accuracy: ", accuracy_rf))

# 랜덤포레스트 활용 예측
rf_predictions <- predict(rf_model2, newdata = test_data)
confusion_matrix_rf <- table(Predicted = rf_predictions, Actual = test_data$MMSE)
confusion_matrix_rf

rf_probabilities <- predict(rf_model2, newdata = test_data, type = "prob")
rf_predictions_prob <- rf_probabilities[, "MCI"]

# 분류 정확도
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy: ", accuracy_rf))

# roc curve
roc_curve <- roc(test_data$MMSE, rf_predictions_prob, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "Random Forest ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_rf["MCI", "MCI"]
TN <- confusion_matrix_rf["N-S", "N-S"]
FP <- confusion_matrix_rf["MCI", "N-S"]
FN <- confusion_matrix_rf["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

################################################################################
#3. 인공신경망

set.seed(123)  # Reproducibility
nn_model <- nnet(MMSE ~ gender + age + edu + marital + religion + como + time + BMI + 
                   interaction + health + restict + smoking + alco + meal + PA + CESD + ADL + IADL, 
                 data = train_data, size = 30, maxit = 200, linout = FALSE)

# 테스트 데이터 예측
nn_predictions <- predict(nn_model, newdata = train_data, type = "class")
confusion_matrix_nn <- table(Predicted = nn_predictions, Actual = train_data$MMSE)

print(confusion_matrix_nn)
accuracy_nn <- sum(diag(confusion_matrix_nn)) / sum(confusion_matrix_nn)
print(paste("Accuracy: ",1- accuracy_nn))

# 검증용 데이터 예측
nn_predictions <- predict(nn_model, newdata = test_data, type = "class")
confusion_matrix_nn <- table(Predicted = nn_predictions, Actual = test_data$MMSE)

nn_probabilities <- predict(nn_model, newdata = test_data, type = "raw")
nn_predictions_prob <- nn_probabilities[,1]

# 혼동행렬
print(confusion_matrix_nn)

# 분류정확도
accuracy_nn <- sum(diag(confusion_matrix_nn)) / sum(confusion_matrix_nn)
print(paste("Accuracy: ",1- accuracy_nn))

# roc curve
roc_curve <- roc(test_data$MMSE, nn_predictions_prob, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "MLP ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_nn["MCI", "MCI"]
TN <- confusion_matrix_nn["N-S", "N-S"]
FP <- confusion_matrix_nn["MCI", "N-S"]
FN <- confusion_matrix_nn["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

##########
# 인공신경망 중요 변수 선정 후 모델 적합(랜포 결과 기준)
set.seed(123)
nn_model2 <- nnet(MMSE ~ age + edu + como + time + BMI + interaction +
                    CESD + meal + IADL+ health, 
                 data = train_data, size = 30, maxit = 200, linout = FALSE)

# 테스트 데이터 예측
nn_predictions <- predict(nn_model2, newdata = train_data, type = "class")
confusion_matrix_nn <- table(Predicted = nn_predictions, Actual = train_data$MMSE)

print(confusion_matrix_nn)
accuracy_nn <- sum(diag(confusion_matrix_nn)) / sum(confusion_matrix_nn)
print(paste("Accuracy: ",1- accuracy_nn))

# 검증용 데이터 예측
nn_predictions <- predict(nn_model2, newdata = test_data, type = "class")
confusion_matrix_nn <- table(Predicted = nn_predictions, Actual = test_data$MMSE)

nn_probabilities <- predict(nn_model2, newdata = test_data, type = "raw")
nn_predictions_prob <- nn_probabilities[,1]

# 혼동행렬
print(confusion_matrix_nn)

# 분류정확도
accuracy_nn <- sum(diag(confusion_matrix_nn)) / sum(confusion_matrix_nn)
print(paste("Accuracy: ", 1- accuracy_nn))

# roc curve
roc_curve <- roc(test_data$MMSE, nn_predictions_prob, levels = c("N-S", "MCI"))
plot(roc_curve, col = "blue", main = "MLP ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_nn["MCI", "MCI"]
TN <- confusion_matrix_nn["N-S", "N-S"]
FP <- confusion_matrix_nn["MCI", "N-S"]
FN <- confusion_matrix_nn["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

################################################################################
#4. SVM

set.seed(123)
svm_model <- svm(MMSE ~ gender + age + edu + marital + religion + como + time + 
                   BMI + interaction + health + restict + smoking + alco + 
                   meal + PA + CESD + ADL + IADL, 
                 data = train_data, kernel = "linear", probability = TRUE) # kernel은 필요한 경우 변경 가능

# 훈련용 데이터 예측
svm_predictions <- predict(svm_model, newdata = train_data)
confusion_matrix_svm <- table(Predicted = svm_predictions, Actual = train_data$MMSE)
print(confusion_matrix_svm)
accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy: ",accuracy_svm))


# 검증용 데이터 예측
svm_predictions <- predict(svm_model, newdata = test_data)

svm_probabilities <- predict(svm_model, newdata = test_data, probability = TRUE)
svm_probabilities <- attr(svm_probabilities, "probabilities")
svm_predictions_prob <- svm_probabilities[, "MCI"]


# 혼동 행렬
confusion_matrix_svm <- table(Predicted = svm_predictions, Actual = test_data$MMSE)
print(confusion_matrix_svm)

# 분류정확도
accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy: ",accuracy_svm))

# roc curve
roc_curve <- roc(test_data$MMSE, svm_predictions_prob, levels = c("N-S", "MCI"))

plot(roc_curve, col = "blue", main = "SVM ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC: ", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_svm["MCI", "MCI"]
TN <- confusion_matrix_svm["N-S", "N-S"]
FP <- confusion_matrix_svm["MCI", "N-S"]
FN <- confusion_matrix_svm["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

##########
# SVM 중요 변수 선정 후 모델 적합(랜덤포레스트 결과 기준)
set.seed(123)
svm_model2 <- svm(MMSE ~ age + edu + como + time + BMI + interaction +
                    CESD + meal + IADL+ health, 
                 data = train_data, kernel = "linear",probability = TRUE) # kernel은 필요한 경우 변경 가능

# 훈련용 데이터 예측
svm_predictions <- predict(svm_model2, newdata = train_data)
confusion_matrix_svm <- table(Predicted = svm_predictions, Actual = train_data$MMSE)
print(confusion_matrix_svm)
accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy: ",accuracy_svm))

# 검증용 데이터 예측
svm_predictions <- predict(svm_model2, newdata = test_data)

svm_probabilities <- predict(svm_model2, newdata = test_data, probability = TRUE)
svm_probabilities <- attr(svm_probabilities, "probabilities")
svm_predictions_prob <- svm_probabilities[, "MCI"]


# 혼동 행렬
confusion_matrix_svm <- table(Predicted = svm_predictions, Actual = test_data$MMSE)
print(confusion_matrix_svm)

# 분류정확도
accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy: ",accuracy_svm))

# roc curve
roc_curve <- roc(test_data$MMSE, svm_predictions_prob, levels = c("N-S", "MCI"))

plot(roc_curve, col = "blue", main = "SVM ROC Curve")

auc_value <- auc(roc_curve)
print(paste("AUC: ", auc_value))

# 혼동 행렬에서 TP, TN, FP, FN 추출
TP <- confusion_matrix_svm["MCI", "MCI"]
TN <- confusion_matrix_svm["N-S", "N-S"]
FP <- confusion_matrix_svm["MCI", "N-S"]
FN <- confusion_matrix_svm["N-S", "MCI"]

# 민감도 (Sensitivity) 계산
sensitivity <- TP / (TP + FN)
print(paste("Sensitivity:", sensitivity))

# 특이도 (Specificity) 계산
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))


#--------------------------------------------------------------------------------
# MODELING                                                                      #
#----------------------------------------------------------------------------------
library(dplyr)
library(e1071)
library(randomForest)
library(pdp)
library(RColorBrewer)
library(tree)
library(PPtreeViz)

setwd("C:\\Users\\User\\Desktop\\graduate\\프랑스")
churn<-read.csv("Churn.csv", header=T, na.strings="NA")
dim(churn)
churn<-churn[complete.cases(churn),]

######## 데이터 전처리
churn$OnlineSecurity <- gsub(" internet service","", churn$OnlineSecurity) 
churn$OnlineBackup <- gsub(" internet service","", churn$OnlineBackup) 
churn$DeviceProtection <- gsub(" internet service","", churn$DeviceProtection) 
churn$TechSupport <- gsub(" internet service","", churn$TechSupport) 
churn$StreamingTV <- gsub(" internet service","", churn$StreamingTV) 
churn$StreamingMovies <- gsub(" internet service","", churn$StreamingMovies) 

churn<- churn %>% mutate(OnlineSecurity=as.factor(OnlineSecurity),
                         OnlineBackup=as.factor(OnlineBackup),
                         DeviceProtection=as.factor(DeviceProtection),
                         TechSupport=as.factor(TechSupport),
                         StreamingTV=as.factor(StreamingTV),
                         StreamingMovies=as.factor(StreamingMovies)
)

churn<-churn %>% select(-customerID)
churn$Churn<-ifelse(churn$Churn=="Yes",1,0)
churn$Churn<-as.factor(churn$Churn)
mean(churn$Churn==1)

######### train, test
set.seed(100)
train_index<-sample(nrow(churn), 0.8*nrow(churn))
train<-churn[train_index,]
test<-churn[-train_index,]

########### 최적 모델
### 1) logistic
glm_step<-glm(Churn ~ SeniorCitizen + Dependents + tenure + MultipleLines + 
                InternetService + OnlineSecurity + OnlineBackup + TechSupport + 
                StreamingTV + StreamingMovies + Contract + PaperlessBilling + 
                PaymentMethod + TotalCharges, data=churn, family="binomial")

pred<-predict(glm_step, churn, type="response")
pred<-ifelse(pred>0.5,1,0)
table(pred, churn$Churn) ## logistic
mean(churn$Churn!=ifelse(pred>0.5,1,0))

### 2) svm
svmfit <- svm(Churn~., data = train, kernel = "radial", 
              cost = 5, 
              gamma =0.04, 
              epsilon = 0.1, scale = FALSE, probability=TRUE)

yprob <- predict(svmfit, test, probability = TRUE)
yprob <- attr(yprob,"probabilities")[,1]

ypred <- predict(svmfit, test, type="response")
table(ypred, test$Churn) ## svm

### 3) randomForest
rf1 <- randomForest(Churn ~. , 
                    data = train,
                    ntree = 500,
                    mtry            = 3,
                    min.node_size = 7,
                    sample.fraction = 0.8,
                    seed            = 123,
                    importance=TRUE)
# PDP
col.pal <- brewer.pal(9,"Blues")
update_geom_defaults("line", list(size = 1.75))
partial(rf1, pred.var = "TotalCharges", plot = TRUE, plot.engine = "ggplot2", prob=TRUE) + theme_bw() +
  theme(axis.title=element_text(size=30), axis.text = element_text(size=15)) +
  labs(y=paste0("probability", "\n"), x=paste0("\n", "TotalCharges"))
partial(rf1, pred.var = "tenure", plot = TRUE, plot.engine = "ggplot2", prob=TRUE) + theme_bw() +
  theme(axis.title=element_text(size=30), axis.text = element_text(size=15)) +
  labs(y=paste0("probability", "\n"), x=paste0("\n", "tenure"))
partial(rf1, pred.var = "MonthlyCharges", plot = TRUE, plot.engine = "ggplot2", prob=TRUE) + 
  theme_bw() + labs(y=paste0("probability", "\n"), x=paste0("\n", "MonthlyCharges")) +
  theme(axis.title=element_text(size=30), axis.text = element_text(size=15))

pred_randomForest <- predict(rf1, test, type = "prob")[,2]
pred_ranfo <- ifelse(pred_randomForest>0.5,1,0)
table(pred_ranfo, test$Churn)
mean(test$Churn!=ifelse(pred_randomForest>0.5,1,0))

### 4) tree - 전체
tr1 <- tree(Churn~., data=churn)
pred_tr <- predict(tr1, churn)[,2] 
pred_tr <- ifelse(pred_tr>0.5,1,0)
table(pred_tr, churn$Churn) # 전체
mean(churn$Churn!=ifelse(pred_tr>0.5,1,0))


### 5) PPtreeViz - 전체
x_train <- scale(model.matrix(Churn~., data=train))[,-1]
y <- train$Churn
train_tbl <- data.frame(cbind(x_train, y))

x_test  <- scale(model.matrix(Churn~.-1, data=test))[,-1]
test_tbl <- data.frame(cbind(x_test, test$Churn))

Tree.result <- PPTreeclass(y~., data=train_tbl, PPmethod="LDA")
pred_pptr <- predict(Tree.result, x_test)
pred_pptr <- ifelse(pred_pptr=="1", "0", "1")
table(pred_pptr, test$Churn)
mean(pred_pptr!=test$Churn)


### 6) DNN
# Libraries
library(tidyverse); library(keras); library(rsample); library(corrr); library(recipes)
library(yardstick)

# test/train
churn$TotalCharges <- log(churn$TotalCharges)
churn$SeniorCitizen <- as.factor(ifelse(churn$SeniorCitizen==1, "Yes", "No"))
churn$Churn <- as.factor(churn$Churn)
churn$gender <- as.factor(churn$gender)
set.seed(100)
train_test_split <- initial_split(churn, prop = 0.8)

train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

# log(TotalCharge)로 변환
train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(
    Churn = Churn %>% as.factor() %>% as.numeric(),
    LogTotalCharges = log(TotalCharges)
  ) %>%
  correlate() %>%
  focus(Churn) %>%
  fashion()

x_train_tbl <- scale(model.matrix(Churn~., data=train_tbl))[,-1]
x_test_tbl  <- scale(model.matrix(Churn~.-1, data=test_tbl))[,-1]

y_train_vec <- train_tbl$Churn
y_test_vec  <- test_tbl$Churn


# DNN 실행
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

summary(model_keras)


# Fit the keras model to the training data
history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 20,
  validation_split = 0.30
)

# Print a summary of the training history
print(history)

plot(history)

model_keras %>% save_model_hdf5("model_keras.hdf5")

model_keras <- load_model_hdf5("model_keras.hdf5")

# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)

# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)

# Precision
tibble(
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate)
)

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)



############## MODEL GRAPHICS ##################
#--------------------------------------------------
dd <- data.frame('loss' = history$metrics$loss,
                 'acc' = history$metrics$acc,
                 'val_loss' = history$metrics$val_loss,
                 'val_acc' = history$metrics$val_acc)



library(plotly)
update_geom_defaults("line", list(size = 2))
ggplot(dd, aes(1:20)) + 
           theme_bw() +
           geom_point(aes(y = acc, color = "train"), size=3) + 
           geom_point(aes(y = val_acc, color = "validation"), size=3) + 
           geom_line(aes(y = acc, color = "train"), size=1.5) +
           geom_line(aes(y = val_acc, color = "validation"), size=1.5) +
  labs(x=paste0("\n","Epoch"), y=paste0("Accuracy","\n")) +
           theme(legend.position="right",
                 legend.justification = c(1,0.5),
                 legend.key.width=unit(2,"line"),
                 legend.background = element_rect(colour = "black", fill = NA),
                 legend.title = element_blank(),
                 legend.text = element_text(size=25),
                 axis.title = element_text(size=30),
                 axis.text = element_text(size=20)
                 ) +
           scale_color_manual(values=c("chartreuse4","blue")) 

ggplot(dd, aes(1:20)) + 
  theme_bw() +
  geom_point(aes(y = loss, color = "train"), size=3) + 
  geom_point(aes(y = val_loss, color = "validation"), size=3) + 
  geom_line(aes(y = loss, color = "train"), size=1.5) +
  geom_line(aes(y = val_loss, color = "validation"), size=1.5) +
  labs(x=paste0("\n","Epoch"), y=paste0("Loss","\n")) +
  theme(legend.position="right",
        legend.justification = c(1,0.5),
        legend.key.width=unit(2,"line"),
        legend.background = element_rect(colour = "black", fill = NA),
        legend.title = element_blank(),
        legend.text = element_text(size=25),
        axis.title = element_text(size=30),
        axis.text = element_text(size=20)
  ) +
  scale_color_manual(values=c("chartreuse4","blue")) 


#########################################
######### 모델  정확도
#########################################

result<-function(a,b,c,d){
  accur<-(a+d)/(a+b+c+d)
  recall<-d/(b+d)
  precision<-d/(c+d)
  F1<-2*(precision*recall)/(precision+recall)
  return (c(accur,recall,precision, F1))
}

# Logistic
result(918,166,129,194)
# SVM
result(936,246,111,114)
# PPtree
result(784,89,263,271)
# Randomforest
result(936,182,111,178)
# DNN
result(925,188,90,203)

#########################################
############ 시각화
#########################################
col=brewer.pal(n=5,"Set1")

# 1) internetservice
ggplot(churn, aes(x=InternetService))+geom_bar()

churn_is<-churn %>% group_by(Churn, InternetService) %>% summarise(count=n())
ggplot(churn_is, aes(x=Churn, y=count)) + 
  geom_bar(stat="identity",aes(fill=InternetService), position=position_dodge()) +
  theme_bw()

percentData <- churn %>% group_by(InternetService) %>% count(Churn) %>%
  mutate(ratio=scales::percent(n/sum(n)))
ggplot(churn, aes(x=InternetService, fill=Churn))+geom_bar(position="fill", width = 0.5)+
  geom_text(data=percentData, aes(y=n,label=ratio),size=3, position=position_fill(vjust=0.5)) +
  scale_x_discrete(limits=c("Fiber optic","DSL","No"))+ylab("ratio")+
  theme_bw()+coord_flip()+scale_fill_brewer()+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))


# 2) onlinesecurity
churn_os<-churn %>% group_by(Churn, OnlineSecurity) %>% summarise(count=n())
ggplot(churn_os, aes(x=Churn, y=count)) + 
  geom_bar(stat="identity",aes(fill=OnlineSecurity), position=position_dodge()) +
  theme_bw()+coord_flip()+scale_fill_brewer(col)

percentData <- churn %>% group_by(OnlineSecurity) %>% count(Churn) %>%
  mutate(ratio=scales::percent(n/sum(n)))
ggplot(churn, aes(x=OnlineSecurity, fill=Churn))+geom_bar(position="fill", width = 0.5)+
  geom_text(data=percentData, aes(y=n,label=ratio), position=position_fill(vjust=0.5)) +
  theme_bw()+coord_flip()+scale_fill_brewer()+ylab("ratio")+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))

# 3) Techsupport
percentData <- churn %>% group_by(TechSupport) %>% count(Churn) %>%
  mutate(ratio=scales::percent(n/sum(n)))
ggplot(churn, aes(x=TechSupport, fill=Churn))+geom_bar(position="fill", width = 0.5)+
  geom_text(data=percentData, aes(y=n,label=ratio), position=position_fill(vjust=0.5)) +
  theme_bw()+coord_flip()+scale_fill_brewer()+ylab("ratio")

# 3-2) PaperlessBilling
summary(churn$PaperlessBilling)
percentData <- churn %>% group_by(PaperlessBilling) %>% count(Churn) %>%
  mutate(ratio=scales::percent(n/sum(n)))
ggplot(churn, aes(x=PaperlessBilling, fill=Churn))+geom_bar(position="fill", width = 0.5)+
  geom_text(data=percentData, aes(y=n,label=ratio), position=position_fill(vjust=0.5)) +
  theme_bw()+coord_flip()+scale_fill_brewer()+ylab("ratio")+
  scale_x_discrete(limits=c("Yes","No"))+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))


# 4) Contract
percentData <- churn %>% group_by(Contract) %>% count(Churn) %>%
  mutate(ratio=scales::percent(n/sum(n)))
ggplot(churn, aes(x=Contract, fill=Churn))+geom_bar(position="fill", width = 0.5)+
  geom_text(data=percentData, aes(y=n,label=ratio), position=position_fill(vjust=0.5)) +
  theme_bw()+coord_flip()+scale_fill_brewer()+ylab("ratio")+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))


# 5) TotalCharges
ggplot(churn, aes(x=Churn, y=TotalCharges, fill=Churn))+geom_boxplot(width=0.5)+
  theme_bw()+coord_flip()+scale_fill_brewer()+scale_x_discrete(limits=c("Yes","No"))+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))


# 6) MonthlyCharges
ggplot(churn, aes(x=Churn, y=MonthlyCharges, fill=Churn))+geom_boxplot(width=0.5)+
  theme_bw()+coord_flip()+scale_fill_brewer()+scale_x_discrete(limits=c("Yes","No"))+
  theme(
    axis.title.x = element_text(size=15),
    axis.title.y = element_text(size=15))

# 7) VIP
vip(rf1, num_features = 9,fill=col)+
  #c("#FFFFFF", "#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506")) + 
  theme_bw() + ggtitle("Variable Importance Plot") +
  theme(plot.title = element_text(size=14, face="bold"))





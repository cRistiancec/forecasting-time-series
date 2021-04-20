library(readr)
plastic <- read.csv(file.choose()) # read the Walmart data
View(plastic) # Seasonality 12 months

# Pre Processing
# input t
plastic["t"] <- c(1:60)
View(plastic)

plastic["t_square"] <- plastic["t"] * plastic["t"]
plastic["log_sales"] <- log(plastic["Sales"])

plot(plastic$Sales,type="o")
# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

plasticsale <- cbind(plastic, X)
colnames(plasticsale)

View(plasticsale)
## Pre-processing completed

attach(plasticsale)

# partitioning
train <- plasticsale[1:50, ]
test <- plasticsale[50:60, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(plasticsale, file = "plasticsales.csv", row.names = F)




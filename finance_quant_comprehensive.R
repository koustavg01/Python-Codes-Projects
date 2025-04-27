
# Comprehensive R Script for Finance and Quantitative Finance

# ----------------------
# 1. Data Loading & Management
# ----------------------

# Load from CSV / Excel
library(readxl)
data <- read.csv("file.csv")
data_excel <- read_excel("file.xlsx")

# Web data
library(quantmod)
getSymbols("AAPL", src = "yahoo")

# Database connection
library(DBI)
library(RSQLite)
con <- dbConnect(RSQLite::SQLite(), "finance.db")
data_db <- dbGetQuery(con, "SELECT * FROM trades")

# Data tidying
library(tidyr)
data_long <- pivot_longer(data, cols = c(price1, price2), names_to = "type", values_to = "price")

# ----------------------
# 2. Time Series & Financial Data Handling
# ----------------------

library(xts)
library(zoo)
ts_data <- xts(data$Price, order.by = as.Date(data$Date))
log_returns <- diff(log(ts_data))
roll_avg <- rollmean(ts_data, k = 20)

# ----------------------
# 3. Descriptive & Exploratory Statistics
# ----------------------

summary(ts_data)
sd(ts_data)
library(e1071)
skewness(ts_data)
kurtosis(ts_data)

library(PerformanceAnalytics)
table.Stats(ts_data)

# ----------------------
# 4. Financial Analysis & Portfolio Management
# ----------------------

# CAPM Model
returns_AAPL <- dailyReturn(AAPL)
returns_SPY <- dailyReturn(SPY)
capm_model <- lm(returns_AAPL ~ returns_SPY)

# Sharpe Ratio
SharpeRatio(returns_AAPL)

# Portfolio Optimization
library(PortfolioAnalytics)
portfolio <- portfolio.spec(assets = colnames(returns))
portfolio <- add.constraint(portfolio, type = "full_investment")
portfolio <- add.objective(portfolio, type = "risk", name = "StdDev")
opt_result <- optimize.portfolio(returns, portfolio, optimize_method = "ROI")

# ----------------------
# 5. Risk Management & Value at Risk
# ----------------------

VaR(returns_AAPL, p = 0.95, method = "historical")
CVaR(returns_AAPL, p = 0.95, method = "historical")

# GARCH Modeling
library(rugarch)
spec <- ugarchspec()
fit <- ugarchfit(spec, returns_AAPL)

# ----------------------
# 6. Time Series & Econometrics Modeling
# ----------------------

library(forecast)
library(tseries)
library(urca)

# ARIMA Model
auto.arima(ts_data)

# Stationarity Tests
adf.test(ts_data)
kpss.test(ts_data)

# Cointegration Test
asset1 <- getSymbols("GOOG", auto.assign = FALSE)
asset2 <- getSymbols("MSFT", auto.assign = FALSE)
cajo_test <- ca.jo(cbind(Cl(asset1), Cl(asset2)), type = "trace", ecdet = "none", K = 2)

# ----------------------
# 7. Quantitative Finance & Algorithmic Trading
# ----------------------

library(quantstrat)
initPortf("myPort", symbols = "AAPL", initDate = "2020-01-01")
initAcct("myAcct", portfolios = "myPort", initDate = "2020-01-01", initEq = 1e6)
initOrders("myPort", initDate = "2020-01-01")

strategy("myStrat", store = TRUE)
add.indicator("myStrat", name = "SMA", arguments = list(x = quote(Cl(mktdata)), n = 20), label = "SMA20")
add.signal("myStrat", name = "sigCrossover", arguments = list(columns = c("Close", "SMA20"), relationship = "gt"), label = "buySignal")

applyStrategy("myStrat", portfolios = "myPort")

# ----------------------
# 8. Machine Learning in Quant Finance
# ----------------------

library(caret)
library(randomForest)
library(xgboost)

# Data splitting
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Random Forest
model_rf <- randomForest(y ~ ., data = trainData)
pred_rf <- predict(model_rf, testData)

# XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(trainData), label = trainData$y)
params <- list(objective = "reg:squarederror")
model_xgb <- xgboost(params = params, data = dtrain, nrounds = 100)

# ----------------------
# 9. Visualization & Reporting
# ----------------------

library(ggplot2)
library(plotly)

# Line plot
ggplot(data, aes(x = Date, y = Price)) + geom_line()

# Candlestick chart
chartSeries(AAPL)

# Interactive plot
plot_ly(x = data$Date, y = data$Price, type = 'scatter', mode = 'lines')

# Risk-return chart
charts.PerformanceSummary(returns_AAPL)

# ----------------------
# 10. Reporting & Dashboards
# ----------------------

# RMarkdown
template <- "---\ntitle: 'Finance Report'\noutput: html_document\n---\n

```{r setup, include=FALSE}
library(quantmod)
```

```{r example}
getSymbols('AAPL')
chartSeries(AAPL)
```
"
writeLines(template, "report.Rmd")

# Shiny Dashboard
library(shiny)
ui <- fluidPage(plotOutput("plot"))
server <- function(input, output) {
  output$plot <- renderPlot({ plot(ts_data) })
}
# shinyApp(ui, server)  # Uncomment to run

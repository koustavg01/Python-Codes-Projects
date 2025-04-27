
# Comprehensive R Script for Finance and Quantitative Finance

# ----------------------
# 1. Data Loading & Management
# ----------------------
# Step 1: Install the 'readxl' package (only once)
install.packages("readxl")

# Step 2: Load the package
library(readxl)

# Step 3: Read the Excel file
data_xlsx <- read_excel("C:/Users/Koustav Goswami/Downloads/sample_financial_data.xlsx")

# Web data
library(quantmod)
# Get stock data 
getSymbols("INFY.NS", src = "yahoo")
# Extract closing prices
closing_prices <- Cl(INFY.NS)
# View result
head(closing_prices)

# Database connection
library(DBI)
library(RSQLite)
con <- dbConnect(RSQLite::SQLite(), "finance.db")
data_db <- dbGetQuery(con, "SELECT * FROM trades")

# Data tidying
# Load necessary library
library(tidyr)

# Use the correct dataset name and columns
data_long <- pivot_longer(data_xlsx, 
                          cols = c(Price, Price2), 
                          names_to = "type", 
                          values_to = "price")

# ----------------------
# 2. Time Series & Financial Data Handling
# ----------------------
library(xts)
library(zoo)

# Convert to time series object using the correct dataframe name
ts_data <- xts(data_xlsx$Price, order.by = as.Date(data_xlsx$Date))

# Calculate log returns
log_returns <- diff(log(ts_data))

# Calculate 20-day rolling average
roll_avg <- rollmean(ts_data, k = 20, fill = NA)

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
returns_WIT <- dailyReturn(WIT)
returns_SPY <- dailyReturn(SPY)
capm_model <- lm(returns_AAPL ~ returns_SPY)

# Sharpe Ratio
SharpeRatio(returns_WIT)

# Portfolio Optimization
library(PortfolioAnalytics)
portfolio <- portfolio.spec(assets = colnames(returns))
portfolio <- add.constraint(portfolio, type = "full_investment")
portfolio <- add.objective(portfolio, type = "risk", name = "StdDev")
opt_result <- optimize.portfolio(returns, portfolio, optimize_method = "ROI")

# ----------------------
# 5. Risk Management & Value at Risk
# ----------------------

VaR(returns_WIT, p = 0.95, method = "historical")
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
# Plot
ggplot(data, aes(x = Date, y = Price)) +
  geom_line()

# Candlestick chart
chartSeries(WIT)

# Interactive plot
plot_ly(x = data$Date, y = data$Price, type = 'scatter', mode = 'lines')

# Risk-return chart
charts.PerformanceSummary(returns_WIT)

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

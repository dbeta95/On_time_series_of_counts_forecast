# Loading packages
library(tsfeatures)
library(gratis) 
library(forecast)

hourly_data = read.csv("hourly_series.csv")
hourly_series = hourly_data[, 2]

my_features <- function(x){
  output <- c(tsfeatures(x))
  output["lambda"] <- BoxCox.lambda(x)
  output["entropy"] <- entropy(x)
  unlist(output)
}


hourly_features = my_features(hourly_series)

set.seed(1)
gen_hourly_100 <- generate_ts_with_target(
  n=100,
  ts.length = 1100,
  freq = 12,
  seasonal = 1,
  features = "my_features",
  selected.features = names(hourly_features),
  target = hourly_features
)
write.csv(gen_hourly_100, file = "hourly_sim_100.csv")
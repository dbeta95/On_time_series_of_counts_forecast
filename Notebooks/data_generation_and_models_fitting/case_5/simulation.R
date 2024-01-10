# Loading packages
library(tsfeatures)
library(gratis) 
library(forecast)

my_features <- function(x){
  output <- c(tsfeatures(x))
  output["lambda"] <- BoxCox.lambda(x)
  output["entropy"] <- entropy(x)
  unlist(output)
}

drivers_killed <- Seatbelts[,"DriversKilled"]

drivers_features <- my_features(drivers_killed)

set.seed(1)
gen_drivers <- generate_ts_with_target(
  n=10,
  ts.length = length(drivers_killed),
  freq = 12,
  seasonal = 1,
  features = "my_features",
  selected.features = names(drivers_features),
  target = drivers_features
)

write.csv(gen_drivers, file = "monthly_sim_10.csv")
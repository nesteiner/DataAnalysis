using MLJFlux
using MLJ
using DataFrames: DataFrame
using Statistics
using Flux
using CSV
using StableRNGs
using Plots

import Random.seed!;
seed!(123)
rng = StableRNG(123)
plotly()

trainData = CSV.read("data/boston-housing/train.csv", DataFrame)
testData = CSV.read("data/boston-housing/test.csv", DataFrame)

columns = [:OverallQual, :GrLivArea, :GarageCars, :TotalBsmtSF, :FullBath, :TotRmsAbvGrd, :YearBuilt]

trainTransformModel = Pipeline(
  FeatureSelector(features = columns),
  dataframe -> coerce(dataframe, Count => Continuous))

processFeature!(dataframe::DataFrame) = begin
  dataframe[!, :GarageCars] = replace(dataframe[!, :GarageCars], "NA" => missing)
  dataframe[!, :GarageCars] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :GarageCars])
  dataframe[!, :TotalBsmtSF] = replace(dataframe[!, :TotalBsmtSF], "NA" => missing)
  dataframe[!, :TotalBsmtSF] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :TotalBsmtSF])

  coerce!(dataframe, Count => Continuous)
  return dataframe
end

testTransformModel = Pipeline(
  FeatureSelector(features = columns),
  processFeature!,
  FillImputer(features = columns),
  # Standardizer(features = columns)
)

trainTransformMach = machine(trainTransformModel, trainData)
testTransformMach = machine(testTransformModel, testData)
fit!(trainTransformMach)
fit!(testTransformMach)

transformedDataTrain = transform(trainTransformMach, trainData)
transformedDataTest = transform(testTransformMach, testData)

X = transformedDataTrain
y = coerce(trainData[!, :SalePrice], Continuous)
train, test = partition(eachindex(y), 0.8, rng=rng)


mutable struct NetworkBuilder <: MLJFlux.Builder
  n1::Int
  n2::Int
  n3::Int
  n4::Int
end

function MLJFlux.build(model::NetworkBuilder, rng, nin, nout)
  init = Flux.glorot_uniform(rng)
  layer1 = Dense(nin, model.n1, init = init)
  layer2 = Dense(model.n1, model.n2, relu, init = init)
  layer3 = Dense(model.n2, model.n3, relu, init = init)
  layer4 = Dense(model.n3, model.n4, relu, init = init)
  layer5 = Dense(model.n4, nout, relu, init = init)
  return Chain(
    layer1,
    layer2,
    layer3,
    layer4,
    layer5
  )
end

regressor = NeuralNetworkRegressor(
  builder = NetworkBuilder(7, 6, 6, 6),
  epochs = 500,
  batch_size = 10,
  lambda = 0.03,
  alpha = 0.4
)

mach = machine(regressor, X, y)
fit!(mach, rows=train)

measure = evaluate!(mach,
                    resampling = CV(nfolds = 6, rng = rng),
                    measure = l1,
                    rows = test)
predictions = predict(mach, transformedDataTest)
output = DataFrame(Id=testData.Id)
output[!, :SalePrice] = predictions
CSV.write("data/boston-housing/submission.csv", output)


# TODO plot learning curve with epoch increment
rangeEpochs = range(regressor, :epochs, lower = 200, upper = 600, scale = :log10)
curve = learning_curve(regressor,
                       X,
                       y,
                       resolution = 10, 
                       range = rangeEpochs,
                       resampling = CV(nfolds = 6, rng = rng),
                       measure = l1)

plot(curve.parameter_values, curve.measurements) |> display
# TODO tuning model with lambda and alpha and batch_size
rangeLambda = range(regressor, :lambda, lower = 0.01, upper = 0.3)
rangeAlpha = range(regressor, :alpha, lower = 0, upper = 1)
grid = Grid(resolution = 3, rng = rng)
cv = CV(nfolds = 6, rng = rng)

tuningModel = TunedModel(model = regressor,
                         tuning = grid,
                         resampling = cv,
                         range = [rangeLambda, rangeAlpha],
                         measure = l1)

tuningMach = machine(tuningModel, X, y)
fit!(tuningMach, rows = train)
measure = evaluate!(tuningMach,
                    rows = test,
                    resampling = cv,
                    measure = l1)

# PROBLEM how to set regularization
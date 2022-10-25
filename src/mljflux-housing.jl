using MLJFlux, MLJ, Statistics, Flux, CSV
using DataFrames: DataFrame, select
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
  dataframe -> coerce(dataframe, Count => Continuous),
  Standardizer(features = columns)
)

processFeature!(dataframe::DataFrame) = begin
  dataframe[!, :GarageCars] = replace(dataframe[!, :GarageCars], "NA" => missing)
  dataframe[!, :GarageCars] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :GarageCars])
  dataframe[!, :TotalBsmtSF] = replace(dataframe[!, :TotalBsmtSF], "NA" => missing)
  dataframe[!, :TotalBsmtSF] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :TotalBsmtSF])

  return coerce(dataframe, Count => Continuous)
end

testTransformModel = Pipeline(
  FeatureSelector(features = columns),
  processFeature!,
  FillImputer(features = columns),
  Standardizer(features = columns)
)

trainTransformMach = machine(trainTransformModel, trainData)
testTransformMach = machine(testTransformModel, testData)
fit!(trainTransformMach)
fit!(testTransformMach)

transformedDataTrain = transform(trainTransformMach, trainData)
transformedDataTest = transform(testTransformMach, testData)

X = transformedDataTrain
y = coerce(trainData[!, :SalePrice], Continuous)
train, test = partition(1:length(y), 0.8, rng=rng)

mutable struct NetworkBuilder <: MLJFlux.Builder
  n1::Int
  n2::Int
  n3::Int
  n4::Int
end

function MLJFlux.build(model::NetworkBuilder, rng, nin, nout)
  init = Flux.glorot_uniform(rng)
  layer1 = Dense(nin, model.n1, relu, init = init)
  layer2 = Dense(model.n1, model.n2, relu, init = init)
  layer3 = Dense(model.n2, model.n3, relu, init = init)
  layer4 = Dense(model.n3, model.n4, relu, init = init)
  layer5 = Dense(model.n4, nout, init = init)
  return Chain(
    layer1,
    layer2,
    layer3,
    layer4,
    layer5
  )
end

regressor = NeuralNetworkRegressor(
  builder = NetworkBuilder(64, 32, 32, 20),
  epochs = 200,
  batch_size = 10,
  lambda = 10,
  alpha = 0.4,
  rng = rng
)

mach = machine(regressor, X, y)
fit!(mach, rows=train; verbosity=2)

measure = evaluate!(mach,
                    resampling = CV(nfolds = 6, rng = rng),
                    measure = l1,
                    rows = test)
predictions = predict(mach, transformedDataTest)
output = DataFrame(Id=testData.Id)
output[!, :SalePrice] = predictions
CSV.write("data/boston-housing/submission.csv", output)
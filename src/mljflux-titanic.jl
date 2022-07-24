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

originData = CSV.read("data/titanic/train.csv", DataFrame)

typeTransformModel!(dataframe::DataFrame) = begin
  if in("Survived", names(dataframe))
    coerce!(dataframe, :Survived => Multiclass)
  end

  coerce!(dataframe, Count => Continuous)
  coerce!(dataframe, Textual => Multiclass)

  return dataframe
end

fillMissingModel = FillImputer(
  features=[:Age, :Embarked],
  continuous_fill = e -> skipmissing(e) |> mode,
  finite_fill = e -> skipmissing(e) |> mode)

newFeatureModel!(dataframe::DataFrame) = begin
  # MODULE FeatureA 聚集 Age, Sex --> 12岁以下儿童以及妇女，12岁以上男性
  feature_filter_a(age, sex) = age >= 12 && sex == "male" ? "A" : "B"
  dataframe[!, :FeatureA] = map(feature_filter_a, dataframe[!, :Age], dataframe[!, :Sex])

  # MODULE FeatureB 聚集 SibSp, Parch ---> 家庭人员数量
  family_size(number) = begin
    if number == 1
      return 0
    elseif number >= 2 && number <= 4
      return 1
    else
      return 2
    end
  end

  dataframe[!, :FeatureB] = map(family_size, dataframe[!, :Parch] .+ dataframe[!, :SibSp] .+ 1)

  # MODULE FeatureC log(Fare + 1), encode(Pclass) -> 1, 2, 3  
  dataframe[!, :Fare] = map(floor, log.(dataframe[!, :Fare] .+ 1))


  # TODO don't forget to coerce scitype
  coerce!(dataframe, :FeatureA => Multiclass, :FeatureB => Continuous)
  return dataframe
end

encodeModel = OneHotEncoder(features=[:Embarked, :FeatureA])
dropUnusedModel = FeatureSelector(features = [:Age, :Sex, :SibSp, :Parch, :Cabin, :PassengerId, :Name, :Ticket], ignore=true)

transformModel = (
  typeTransformModel!,
  fillMissingModel,
  newFeatureModel!,
  encodeModel,
  dropUnusedModel
)
transformMachine = machine(transformModel, originData)

fit!(transformMachine)
outputData = MLJ.transform(transformMachine, originData)


originSample = CSV.read("data/titanic/test.csv", DataFrame)
# generic typeTransformModel, ignore
fillMissingModel = FillImputer(features=[:Age, :Fare], continuous_fill = e -> skipmissing(e) |> mode)

# generic new feature generate
# generic encode model
# generic drop unused
transformSampleModel = Pipeline(
  typeTransformModel!,
  fillMissingModel,
  newFeatureModel!,
  encodeModel,
  dropUnusedModel)

transformSampleMachine = machine(transformSampleModel, originSample)
fit!(transformSampleMachine)

outputSample = MLJ.transform(transformSampleMachine, originSample)

Y, X = unpack(outputData, colname -> colname == :Survived, colname -> true)
rng = StableRNG(1234)
trainRow, testRow = partition(eachindex(Y), 0.7, rng=rng)

mutable struct NetworkBuilder <: MLJFlux.Builder
  n1::Int
  n2::Int
  n3::Int
  n4::Int
end

function MLJFlux.build(model::NetworkBuilder, rng, nin, nout)
  init = Flux.glorot_uniform(rng)
  return Chain(
    Dense(nin, model.n1, relu, init = init),
    Dense(model.n1, model.n2, relu, init = init),
    Dense(model.n2, model.n3, relu, init = init),
    Dense(model.n3, model.n4, relu, init = init),
    Dense(model.n4, nout, relu, init = init)
  )
end

classifier = NeuralNetworkClassifier(
  builder = NetworkBuilder(10, 6, 6, 6),
  finaliser = softmax,
  epochs = 200,
  batch_size = 10,
  lambda = 0.01,
  alpha = 0.4
)

mach = machine(classifier, X, Y)
fit!(mach, rows = trainRow)

measure = evaluate!(mach,
                    resampling = CV(nfolds = 6, rng = rng),
                    measure = cross_entropy,
                    rows = testRow)

outputPredict = mode.(predict(mach, outputSample)) |> nums -> convert(Vector{Int}, nums)

output_frame = DataFrame()
output_frame[!, :PassengerId] = convert(Vector{Int}, originSample[!, :PassengerId])
output_frame[!, :Survived] = outputPredict
CSV.write("data/titanic/predict.csv", output_frame)
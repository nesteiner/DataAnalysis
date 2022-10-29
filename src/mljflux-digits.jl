using MLJFlux, Flux, StableRNGs, MLDatasets, MLJ, CSV
using Flux: onehotbatch
using DataFrames: DataFrame

# load data
function transformDataType!(dataframe::DataFrame)
  coerce!(dataframe, Count => Continuous)
  if in(:label, names(dataframe))
    coerce!(dataframe, :label => Multiclass)
  end

  return dataframe
end


function loaddata(path::AbstractString)
  origindata = CSV.read(path, DataFrame)
  transformDataType!(origindata)
  y, X = unpack(origindata, colname -> colname == :label, colname -> true)
  images = reshape(transpose(Matrix(X)) ./ 255.0, (28, 28, :))

  labels = coerce(y, Multiclass)
  images = coerce(images, GrayImage)

  return labels, images
end

function loadtestdata(path::AbstractString)
  testdata = CSV.read(path, DataFrame)
  transformDataType!(testdata)
  images = reshape(transpose(Matrix(testdata)) ./ 255.0, (28, 28, :))
  images = coerce(images, GrayImage)
  return images
end

include("lenet5.jl")
include("alexnet.jl")

function makepredict(pathtrain::AbstractString, pathtest::AbstractString, pathsubmission::AbstractString)
  rng = StableRNG(1234)
  y, X = loaddata(pathtrain)
  # model = buildLeNet(batchsize = 32, epochs = 1, lambda = 10.0f0, alpha = 0.5f0)
  model = buildAlexNet(batchsize = 32, epochs = 1, lambda = 10.0f0, alpha = 0.5f0, usegpu = false)
  mach = machine(model, X, y)
  fit!(mach; verbosity = 2)

  testdata = loadtestdata(pathtest)
  output = map(x -> convert(Int, x), mode.(predict(mach, testdata)))

  outputdataframe = DataFrame()
  outputdataframe[!, :ImageId] = 1:length(output);
  outputdataframe[!, :Label] = output
  CSV.write(pathsubmission, outputdataframe)
end

makepredict("data/digits-recognizer/train.csv", 
            "data/digits-recognizer/test.csv", 
            "data/digits-recognizer/submission.csv")
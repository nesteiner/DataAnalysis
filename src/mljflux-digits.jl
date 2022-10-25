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

function makepredict(pathtrain::AbstractString, pathtest::AbstractString, pathsubmission::AbstractString)
  rng = StableRNG(1234)
  y, X = loaddata(pathtrain)
  # trainrow, testrow = partition(eachindex(y), 0.7, rng = rng)
  model = buildmodel()
  mach = machine(model, X, y)
  fit!(mach; verbosity = 2)

  testdata = loadtestdata(pathtest)
  output = map(x -> convert(Int, x), mode.(predict(mach, testdata)))

  outputdataframe = DataFrame()
  outputdataframe[!, :ImageId] = 1:length(output);
  outputdataframe[!, :Label] = output
  CSV.write(pathsubmission, outputdataframe)
end

mutable struct LeNet5 <: MLJFlux.Builder
  filtersize::Int
  channels1::Int
  channels2::Int
  channels3::Int
end

function MLJFlux.build(lenet5::LeNet5, rng, nin, nout, nchannels)
  @info nin
  k, c1, c2, c3 = lenet5.filtersize, lenet5.channels1, lenet5.channels2, lenet5.channels3
  mod(k, 2) == 1 || error("filter size must be odd")
  p = div(k - 1, 2)
  init = Flux.glorot_uniform(rng)

  front = Chain(
    Conv((k, k), nchannels => c1, pad = (p, p), relu, init = init),
    MaxPool((2, 2)),
    Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
    MaxPool((2, 2)),
    Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
    MaxPool((2, 2)),
    x -> reshape(x, (:, last(size(x))))
  )

  d = Flux.outputsize(front, (nin..., nchannels, 1)) |> first
  return Chain(front, Dense(d, nout, init = init))
end

function buildmodel()
  classifier = ImageClassifier(
    builder = LeNet5(5, 16, 32, 32),
    batch_size = 50,
    epochs = 1,
    rng = StableRNG(1234),
    lambda = 0.01,
    alpha = 0.4
  )
  return classifier
end

struct Net end

function MLJFlux.build(net::Net, rng, nin, nout, nchannels)
    array = (nin..., nchannels, 32)
    # this is ok
    return @autosize (nin..., nchannels, 32) Chain(
      Conv((5, 5), _ => 6, relu),
      MaxPool((2, 2)),
      Conv((5, 5), _ => 16, relu),
      MaxPool((2, 2)),
      Flux.flatten,
      Dense(_ => 120, relu),
      Dense(120 => 84, relu),
      Dense(84 => nout)
    )
end

function buildmodel()
  return ImageClassifier(
    builder = Net(),
    batch_size = 32,
    epochs = 5,
    rng = StableRNG(1234),
    lambda = 0.01,
    alpha = 0.4
  )
end

makepredict("data/digits-recognizer/train.csv", "data/digits-recognizer/test.csv", "data/digits-recognizer/submission.csv")
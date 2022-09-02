using MLJFlux, MLJ, Flux, DataFrames, CSV, StableRNGs

origindata = CSV.read("data/digits-recognizer/train.csv", DataFrame)

mutable struct NetworkBuilder <: MLJFlux.Builder
  n1::Int
  n2::Int
  n3::Int
  n4::Int
  n5::Int
end

function MLJFlux.build(model::NetworkBuilder, rng, nin, nout)
  init = Flux.glorot_uniform(rng)
  return Chain(
    Dense(nin, model.n1, relu, init = init),
    Dense(model.n1, model.n2, relu, init = init),
    Dense(model.n2, model.n3, relu, init = init),
    Dense(model.n3, model.n4, relu, init = init),
    Dense(model.n4, model.n5, relu, init = init),
    Dense(model.n5, nout, relu, init = init)
  )
end


function transformDataType!(dataframe::DataFrame)
  coerce!(dataframe, Count => Continuous)
  coerce!(dataframe, :label => Multiclass)

  return dataframe
end

rng = StableRNG(1234)
classifier = NeuralNetworkClassifier(
  lambda = 0.01,
  builder = NetworkBuilder(10, 8, 8, 8, 6),
  batch_size = 10,
  epochs = 600,
  alpha = 0.4,
  rng = rng,
  acceleration = OpenCLLibs()
)

transformDataType!(origindata)
y, X = unpack(origindata, colname -> colname == :label, colname -> true)
trainrow, testrow = partition(eachindex(y), 0.7, rng = rng)

mach = machine(classifier, X, y)
fit!(mach, rows = trainrow)

measure = evaluate!(mach, rows = testrow,
                    resampling = CV(nfolds = 6, rng = rng),
                    measure = cross_entropy)

testdata = CSV.read("data/digits-recognizer/test.csv", DataFrame)
output = map(x -> convert(Int, x), mode.(predict(mach, testdata)))

outputdataframe = DataFrame()
outputdataframe[!, :ImageId] = 1:length(output);
outputdataframe[!, :Label] = output
CSV.write("data/digits-recognizer/submission.csv", outputdataframe)

# save model
import Serialization
Serialization.serialize("model/digits-recognizer", mach)
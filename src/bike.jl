using MLJFlux, Flux, MLJ, DataFrames, CSV, StatsBase
using Plots, StatsPlots
using Dates
plotly(size = (1600, 1200))

origindata = CSV.read("data/bike-sharing/train.csv", DataFrame)

function transformDataType!(dataframe::DataFrame)
  # 转换日期
  datetimes = map(x -> DateTime(x, "yyyy-mm-dd HH:MM:SS"), dataframe[!, :datetime])

  years = map(year, datetimes)
  months = map(month, datetimes)
  days = map(day, datetimes)
  weekdays = map(dayofweek, datetimes)
  hours = map(hour, datetimes)
  
  dataframe[!, :year] = years
  dataframe[!, :month] = months
  dataframe[!, :weekday] = weekdays
  dataframe[!, :day] = days
  dataframe[!, :hour] = hours


  # 转换季节，工作日，天气
  seasonmap = Dict(1 => "Spring", 2 => "Summer", 3 => "Autumn", 4 => "Winter")
  holidaymap = Dict(1 => "Workingday", 0 => "Weekday")
  weathermap = Dict(1 => "Sunny", 2 => "Cloudy", 3 => "Flurry", 4 => "Heavy Snow")
  dataframe[!, :season] = map(x -> seasonmap[x], dataframe[!, :season])
  dataframe[!, :workingday] = map(x -> holidaymap[x], dataframe[!, :workingday])
  dataframe[!, :weather] = map(x -> weathermap[x], dataframe[!, :weather])
  return dataframe
end

transformDataType!(origindata)

function plotHeatmap(dataframe::DataFrame)
  _schema = schema(dataframe)
  columns = collect(_schema.names)
  scitypes = collect(_schema.scitypes)
  columns = columns[scitypes .!= Textual]

  cormatrix = cor(Matrix(select(dataframe, columns)))
  heatmap(string.(columns), string.(columns), cormatrix) |> display
end

plotHeatmap(origindata)
# MODULE 数据可视化
function plotYearAndCount(dataframe::DataFrame)
  groupDataframe = groupby(dataframe, :year)

  years = Int[]
  counts = Int[]
  for _dataframe in groupDataframe
    year = first(_dataframe[!, :year])
    count = reduce(+, _dataframe[!, :count])

    push!(years, year)
    push!(counts, count)
  end

  bar(years, counts) |> display
end

plotYearAndCount(origindata)

function plotPieOfCount(dataframe::DataFrame)
  totalcount = reduce(+, dataframe[!, :count])
  casualcount = reduce(+, dataframe[!, :casual])
  registeredcount = reduce(+, dataframe[!, :registered])

  xs = ["Casual", "Registered"]
  ys = [casualcount / totalcount, registeredcount / totalcount]

  pie(xs, ys) |> display
end

plotPieOfCount(origindata)

function plotTimeAndCount(dataframe::DataFrame, feature::Symbol)
  features = [:hour, feature]
  groupDataframe = groupby(dataframe, features)
  hours = Int[]
  counts = Float64[]
  features = []
  for _dataframe in groupDataframe
    hour = first(_dataframe[!, :hour])
    count = mean(_dataframe[!, :count])
    _feature = first(_dataframe[!, feature])
    push!(hours, hour)
    push!(counts, count)
    push!(features, _feature)
  end

  partcount = 24
  p = plot()
  for (_hours, _counts, labels) in Iterators.zip(Iterators.partition(hours, partcount),
                                                 Iterators.partition(counts, partcount),
                                                 Iterators.partition(features, partcount))
    
    indexs = sortperm(_hours)
    xs = _hours[indexs]
    ys = _counts[indexs]
    
    plot!(p, xs, ys, label = first(labels))
  end

  display(p)
end

plotTimeAndCount(origindata, :season)
plotTimeAndCount(origindata, :workingday)
plotTimeAndCount(origindata, :weather)

boxplot(origindata[!, :count]) |> display
boxplot(origindata[!, :hour], origindata[!, :count]) |> display
boxplot(origindata[!, :weekday], origindata[!, :count]) |> display

# DONE plot weather and count
boxplot(origindata[!, :weather], origindata[!, :count]) |> display
# DONE plot season and count
boxplot(origindata[!, :season], origindata[!, :count]) |> display
# TODO plot weather, season and count
# ATTENTION holy shit is this !


# MODULE 预测

function fetchTransformedTrainData(traindata::DataFrame)
  function transformDateTime!(dataframe::DataFrame)
    datetimes = map(x -> DateTime(x, "yyyy-mm-dd HH:MM:SS"), dataframe[!, :datetime])

    years = map(year, datetimes)
    months = map(month, datetimes)
    days = map(day, datetimes)
    weekdays = map(dayofweek, datetimes)
    hours = map(hour, datetimes)
    
    dataframe[!, :year] = years
    dataframe[!, :month] = months
    dataframe[!, :weekday] = weekdays
    dataframe[!, :day] = days
    dataframe[!, :hour] = hours

    return dataframe
  end

  featureSelector = FeatureSelector(
    features = [:datetime, :casual, :registered],
    ignore = true
  )
  onehotEncoder = OneHotEncoder(
    features = [:season, :holiday, :workingday, :weather]
  )

  function coerceCount!(dataframe::DataFrame)
    coerce!(dataframe, Count => Continuous)
    return dataframe
  end

  transformModel = Pipeline(
    transformDateTime!,
    featureSelector,
    onehotEncoder,
    coerceCount!
  )

  transformMachine = machine(transformModel, traindata)
  fit!(transformMachine)
  # TODO 转换 traindata testdata
  transformedTrainData = MLJ.transform(transformMachine, copy(traindata))
  return transformedTrainData
end

function fetchTransformedTestData(testdata::DataFrame)
  function transformDateTime!(dataframe::DataFrame)
    datetimes = map(x -> DateTime(x, "yyyy-mm-dd HH:MM:SS"), dataframe[!, :datetime])

    years = map(year, datetimes)
    months = map(month, datetimes)
    days = map(day, datetimes)
    weekdays = map(dayofweek, datetimes)
    hours = map(hour, datetimes)
    
    dataframe[!, :year] = years
    dataframe[!, :month] = months
    dataframe[!, :weekday] = weekdays
    dataframe[!, :day] = days
    dataframe[!, :hour] = hours

    return dataframe
  end

  featureSelector = FeatureSelector(
    features = [:datetime],
    ignore = true
  )

  function coerceCount!(dataframe::DataFrame)
    coerce!(dataframe, Count => Continuous)
    return dataframe
  end

  transformModel = Pipeline(
    transformDateTime!,
    featureSelector,
    onehotEncoder,
    coerceCount!
  )

  transformMachine = machine(transformModel, testdata)
  fit!(transformMachine)
  transformedTestData  = MLJ.transform(transformMachine, copy(testdata))
  return transformedTestData
end

traindata = CSV.read("data/bike-sharing/train.csv", DataFrame)
testdata = CSV.read("data/bike-sharing/test.csv", DataFrame)

transformedTrainData = fetchTransformedTrainData(traindata)
transformedTestData = fetchTransformedTestData(testdata)

# TODO MLJFlux prediction
using MLJFlux, Flux, StableRNGs
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

function predictOutput(mach::Machine, inputtest::DataFrame)
  output = MLJ.predict(mach, inputtest)
  outputdataframe = DataFrame()
  outputdataframe[!, :datetime] = testdata[!, :datetime]
  outputdataframe[!, :count] = output
  CSV.write("data/bike-sharing/submissing.csv", outputdataframe)

end

# TODO make function but not global data
# TODO plot origin y and predict y
function plotPrediction(dataframe::DataFrame, output::Vector)
  difference = output .- dataframe[!, :count]
  plot(difference) |> display
end


# ATTENTION use function to get data, but not model
rng = StableRNG(1234)
regressor = NeuralNetworkRegressor(
  lambda = 0.01,
  builder = NetworkBuilder(10, 8, 6, 6),
  batch_size = 5,
  epochs = 600,
  alpha = 0.4,
  rng = rng
)

y, X = unpack(transformedTrainData, colname -> colname == :count, colname -> true)
trainrow, testrow = partition(eachindex(y), 0.7, rng = rng)
regressor = machine(regressor, X, y)
fit!(regressor, rows = trainrow)

measure = evaluate!(regressor,
                    resampling = CV(nfolds = 6, rng = rng),
                    measure = [l1, l2],
                    rows = testrow)


columns = names(transformedTrainData)
columns = columns[columns .!= "count"]
plotPrediction(transformedTrainData, MLJ.predict(regressor, select(transformedTrainData, columns)))

predictOutput(regressor, transformedTestData)
using MLJFlux, Flux, MLJ, DataFrames, CSV, StatsBase, Dates
import JSON

using Plots, StatsPlots
plotly()
figuresize = (1600, 1200)
origindata = CSV.read("data/telco-customer-churn/data.csv", DataFrame)

# MODULE 数据清洗
function findNullData(dataframe::DataFrame)
  for column in names(dataframe)
    missingcount = count(ismissing, dataframe[!, column])
    println("$column: \t $missingcount")
  end
end

findNullData(origindata)
schema(origindata)

function transformTotalCharges(dataframe::DataFrame)
  indexs = dataframe[!, :TotalCharges] .== " "
  dataframe[!, :TotalCharges][indexs] .= string.(dataframe[!, :MonthlyCharges][indexs])

  dataframe[!, :TotalCharges] = map(x -> parse(Float64, x), dataframe[!, :TotalCharges])
  dataframe[!, :tenure][indexs] .= 1
  return dataframe
end

transformTotalCharges(origindata)

# MODULE 可视化分析
function plotChurn(dataframe::DataFrame)
  counts = countmap(dataframe[!, :Churn])

  yescount = counts["Yes"]
  nocount = counts["No"]
  total = yescount + nocount

  xs = ["Yes", "No"]
  ys1 = [yescount / total, nocount / total]
  ys2 = [yescount, nocount]
  pie(xs, ys1, aspect_ratio = :equal) |> display

  bar(xs, ys2, size = figuresize) |> display 
end

plotChurn(origindata)

# DONE 用户属性分析
function plotPercentages(dataframe::DataFrame, feature::Symbol, ymatrix::Matrix{Float64})
  columns = [feature, :Churn]
  groupDataframe = groupby(select(dataframe, columns), feature)
  xs = []

  let
    index = 1
    for _dataframe in groupDataframe
      x = first(_dataframe[!, feature])
      push!(xs, x)

      yescount = count(isequal("Yes"), _dataframe[!, :Churn])
      nocount = count(isequal("No"), _dataframe[!, :Churn])
      total = yescount + nocount

      ymatrix[index, :] = [yescount / total, nocount / total]
      index += 1
    end
  end
  groupedbar(ymatrix, xticks = (1:length(xs), xs), label = ["Yes" "No"]) |> display
end

plotPercentages(origindata, :SeniorCitizen, ones((2, 2)))
plotPercentages(origindata, :gender, ones((2, 2)))
plotPercentages(origindata, :Partner, ones((2, 2)))
plotPercentages(origindata, :Dependents, ones((2, 2)))

density(origindata.tenure, group = origindata.Churn, size = figuresize) |> display
# DONE 服务属性分析
plotPercentages(origindata, :MultipleLines, ones((3, 2)))
plotPercentages(origindata, :InternetService, ones((3, 2)))

function plotPaperlessBillingChurn(dataframe::DataFrame)
  columns = [:PaperlessBilling, :Contract, :Churn]
  groupDataframe = groupby(select(dataframe, columns), :PaperlessBilling)
  array = unique(dataframe[!, :Contract])
  for _dataframe in groupDataframe
    _dataframe = filter(row -> row.Churn == "Yes", _dataframe)
    paperlessbilling = first(_dataframe[!, :PaperlessBilling])

    churn1 = count(isequal(array[1]), _dataframe[!, :Contract])
    churn2 = count(isequal(array[2]), _dataframe[!, :Contract])
    churn3 = count(isequal(array[3]), _dataframe[!, :Contract])

    total = churn1 + churn2 + churn3
    ys = [churn1 / total, churn2 / total, churn3 / total]
    bar(array, ys, title = "PaperlessBilling = $paperlessbilling") |> display
  end
  
end

plotPaperlessBillingChurn(origindata)

function plotNumberOfCustomer(dataframe::DataFrame)
  columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
  ymatrix = ones((length(columns), 3))

  index = 1
  for column in columns
    _dataframe = select(filter(row -> row.InternetService != "No", dataframe), [column, "Churn"])
    
    count1 = count(isequal("Yes"), _dataframe[!, column])
    count2  = count(isequal("No"),  _dataframe[!, column])
    if column != "MultipleLines"
      ymatrix[index, :] = [count1, count2, 0]
    else
      ymatrix[index, :] = [count1, count2, count(isequal("No phone service"), _dataframe[!, column])]
    end
    index += 1
    
  end

  groupedbar(ymatrix, xticks = (1:length(columns), columns), label = ["Has Service" "No Service" "No Service"], size = figuresize) |> display
end

plotNumberOfCustomer(origindata)

function plotNumberOfChurnCustomer(dataframe::DataFrame)
  columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
  ymatrix = ones((length(columns), 2))

  index = 1
  for column in columns
    _dataframe = select(filter(row -> row.InternetService != "No" && row.Churn == "Yes", dataframe), [column, "Churn"])
    # has service but churn
    yescount = count(isequal("Yes"), _dataframe[!, column])
    # has no service but churn
    nocount = count(isequal("No"), _dataframe[!, column])
    ymatrix[index, :] = [yescount, nocount]
    index += 1
  end

  groupedbar(ymatrix, xticks = (1:length(columns), columns), label = ["Has Service" "No Service"], size = figuresize) |> display
end

plotNumberOfChurnCustomer(origindata)

# DONE 合同属性分析
plotPercentages(origindata, :PaymentMethod, ones((4, 2)))

density(origindata.MonthlyCharges, group = origindata.Churn, size = figuresize) |> display
density(origindata.TotalCharges, group = origindata.Churn, size = figuresize) |> display
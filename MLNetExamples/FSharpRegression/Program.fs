open System
open Microsoft.ML
open Microsoft.ML.Data

let trainFile = "./SalaryTrain.csv"
let testFile = "./SalaryTest.csv"

type SalaryData = {
  [<LoadColumn(0)>]
  YearsExperience: float32

  [<LoadColumn(1)>]
  Salary: float32
}

[<EntryPoint>]
let main argv =
    let context = MLContext()

    let trainData = context.Data.LoadFromTextFile<SalaryData>(trainFile, hasHeader=true, separatorChar=',')
    let testData = context.Data.LoadFromTextFile<SalaryData>(testFile, hasHeader=true, separatorChar=',')

    // let trainData = loadDataByPath trainFile
    //let testData = loadDataByPath testFile

    let pipeline = 
      EstimatorChain()
        .Append(context.Transforms.Concatenate("Features", "YearsExperience"))
        .Append(context.Transforms.CopyColumns(("Label", "Salary")))
        .Append(context.Regression.Trainers.Sdca("Label", "Features"))

    printfn "Training model..."
    let model = pipeline.Fit trainData

    let predictions = model.Transform testData

    printfn "Evaluating model..."
    let metrics = context.Regression.Evaluate(predictions, "Label", "Score")

    printfn "RMS - %.2f" metrics.RootMeanSquaredError

    Console.ReadLine() |> ignore

    0
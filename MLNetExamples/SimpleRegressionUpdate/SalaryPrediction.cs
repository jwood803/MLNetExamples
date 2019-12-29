using Microsoft.ML.Data;

namespace SimpleRegressionUpdate
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}
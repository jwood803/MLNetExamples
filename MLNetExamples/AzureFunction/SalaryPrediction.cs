using Microsoft.ML.Data;

namespace AzureFunction
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}

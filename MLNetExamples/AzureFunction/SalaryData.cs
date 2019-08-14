using Microsoft.ML.Data;

namespace AzureFunction
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience { get; set; }

        [LoadColumn(1)]
        public float Salary { get; set; }
    }

}

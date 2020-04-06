using Microsoft.ML.Data;

namespace LargeFeatures
{
    internal class SensorData
    {
        [LoadColumn(0)]
        public float ExperienceNumber { get; set; }

        [LoadColumn(1)]
        public string Batch { get; set; }

        [LoadColumn(2)]
        public float AcetoneConcentration { get; set; }

        [LoadColumn(3)]
        public float EthanolConcentration { get; set; }

        [LoadColumn(4)]
        public string Gas { get; set; }

        [LoadColumn(5)]
        public string Lab { get; set; }

        [LoadColumn(6)]
        public string ColorCode { get; set; }

        [LoadColumn(7, 431)]
        [VectorType(431)]
        public float[] Features { get; set; }
    }
}
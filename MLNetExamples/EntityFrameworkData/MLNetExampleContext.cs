using Microsoft.EntityFrameworkCore;

namespace EntityFrameworkData
{
    public partial class MLNetExampleContext : DbContext
    {
        public DbSet<SalaryData> Salaries { get; set; }

        public MLNetExampleContext(DbContextOptions<MLNetExampleContext> options)
            : base(options)
        {
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                optionsBuilder.UseSqlServer("Server=(localdb)\\mssqllocaldb;Database=EFProviders.InMemory;Trusted_Connection=True;ConnectRetryCount=0");
            }
        }
    }
}

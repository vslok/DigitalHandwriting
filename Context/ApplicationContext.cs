using DigitalHandwriting.Models;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace DigitalHandwriting.Context
{

    public class ApplicationContext : DbContext
    {
        public ApplicationContext()
        {
            Database.EnsureCreated();
        }
        public DbSet<User> Users { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder builder)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            if (baseDir.Contains("bin"))
            {
                int index = baseDir.IndexOf("bin");
                baseDir = baseDir.Substring(0, index);
            }

            var connectionStringBuilder = new SqliteConnectionStringBuilder { DataSource = Path.Combine(baseDir, ConstStrings.DbName) }; // Use Path.Combine
            var connectionString = connectionStringBuilder.ToString();

            // For SQLite with SQLCipher, connection handling might need to be careful if you intend to keep it open.
            // Typically, EF Core manages connection opening/closing unless you provide an already open connection.
            // If you are providing the connection string, EF will handle opening it.
            // var connection = new SqliteConnection(connectionString);
            // connection.Open(); // If you open it, you are responsible for its lifetime or EF might complain.
            // using var command = connection.CreateCommand();
            // command.CommandText = "PRAGMA key = 'testpragma';"; // This needs to be done AFTER connection is opened but BEFORE EF uses it.
            // command.ExecuteNonQuery();

            builder.UseSqlite(connectionString); // Pass the connection string directly
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder); // Good practice to call the base method

            modelBuilder.Entity<User>(entity =>
            {
                // Configure HSampleValues to be stored as a JSON string
                entity.Property(u => u.HSampleValues)
                    .HasConversion(
                        v => JsonSerializer.Serialize(v, (JsonSerializerOptions)null), // Convert List<List<double>> to string
                        v => JsonSerializer.Deserialize<List<List<double>>>(v, (JsonSerializerOptions)null)
                             ?? new List<List<double>>() // If DB string is null, deserialize to new empty list
                    );

                // Configure UDSampleValues to be stored as a JSON string
                entity.Property(u => u.UDSampleValues)
                    .HasConversion(
                        v => JsonSerializer.Serialize(v, (JsonSerializerOptions)null),
                        v => JsonSerializer.Deserialize<List<List<double>>>(v, (JsonSerializerOptions)null)
                            ?? new List<List<double>>()
                    );

                // Add other User entity configurations here if needed in the future
                // e.g., entity.HasKey(u => u.Id);
                // e.g., entity.Property(u => u.Login).IsRequired(); (though [Required] attribute does this)
            });

            // If you have other entities, configure them here
            // modelBuilder.Entity<OtherEntity>(entity => { ... });
        }
    }
}

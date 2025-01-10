using DigitalHandwriting.Models;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Data.Common;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

            var connectionStringBuilder = new SqliteConnectionStringBuilder { DataSource = $"{baseDir}{ConstStrings.DbName}" };
            var connection = new SqliteConnection(connectionStringBuilder.ToString());

            // Set up the passphrase for SQLCipher encryption
            connection.Open();

            /*using var command = connection.CreateCommand();
            command.CommandText = "PRAGMA key = 'testpragma';";
            command.ExecuteNonQuery();*/

            builder.UseSqlite(connection);
        }
    }
}

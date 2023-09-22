using DigitalHandwriting.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Context
{

    public class ApplicationContext : DbContext
    {
        private static bool _created = true;
        public ApplicationContext()
        {
            if (!_created)
            {
                _created = true;
                Database.EnsureDeleted();
                Database.EnsureCreated();
            }
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

            builder.UseSqlite($"Data source={baseDir}{ConstStrings.DbName}");
        }
    }
}

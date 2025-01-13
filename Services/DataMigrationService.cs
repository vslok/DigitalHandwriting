using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using CsvHelper.Configuration;
using CsvHelper;
using DigitalHandwriting.Converters;
using System.Globalization;
using System.Text.Json;

namespace DigitalHandwriting.Services
{
    internal class CsvImportUser
    {
        public string Subject { get; set; }
        public int SessionIndex { get; set; }
        public int Rep { get; set; }
        public string Password { get; set; }
        public double[] H { get; set; }
        public double[] DD { get; set; }
        public double[] UD { get; set; }
    }

    public class DataMigrationService
    {
        private readonly UserRepository _userRepository;

        public DataMigrationService()
        {
            _userRepository = new UserRepository();
        }

        public void ImportDataFromCsv(string filePath)
        {
            var records = this.ReadUsersFromCsv(filePath);
            foreach (var record in records)
            {
                _userRepository.AddUser(new User()
                {
                    Login = record.Subject,
                    Password = EncryptionService.GetPasswordHash(record.Password, out var salt),
                    Salt = salt,
                    KeyPressedTimes = JsonSerializer.Serialize(record.H),
                    BetweenKeysTimes = JsonSerializer.Serialize(record.UD),
                    BetweenKeysPressTimes = JsonSerializer.Serialize(record.DD),
                });
            }
        }

        private List<CsvImportUser> ReadUsersFromCsv(string filePath)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());
                var records = csv.GetRecords<CsvImportUser>().ToList();
                return records;
            }
        }
    }
}

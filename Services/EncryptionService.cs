using Konscious.Security.Cryptography;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    static class EncryptionService
    {
        public static string GetPasswordHash(string password, out string salt)
        {
            salt = Guid.NewGuid().ToString();
            return GeneratePasswordHash(password, salt);
        }

        public static string GetPasswordHash(string password, string salt) => GeneratePasswordHash(password, salt);

        private static string GeneratePasswordHash(string password, string salt)
        {
            var argon2 = new Argon2i(Encoding.UTF8.GetBytes(password));
            argon2.DegreeOfParallelism = Environment.ProcessorCount;
            argon2.MemorySize = 8192;
            argon2.Iterations = 40;
            argon2.Salt = Encoding.UTF8.GetBytes(salt);

            var hash = argon2.GetBytes(128);
            var temp = Encoding.UTF8.GetString(hash, 0, hash.Length);
            return Encoding.UTF8.GetString(hash, 0, hash.Length);
        }
    }
}

using DigitalHandwriting.Context;
using DigitalHandwriting.Models;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Repositories
{
    public class UserRepository
    {
        private readonly ApplicationContext db = new ApplicationContext();

        public UserRepository() { }

        public void AddUser(User user)
        {
            db.Users.Add(user);
            db.SaveChanges();
        }

        public User? GetUser(string userLogin)
        {
            return db.Users.Select(user => user).Where(user => user.Login == userLogin).FirstOrDefault();
        }

        public List<User> getAllUsers()
        {
            return db.Users.ToList();
        }
    }
}

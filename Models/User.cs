using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DigitalHandwriting.Models
{
    [PrimaryKey("Id")]
    public class User
    {
        public int Id { get; }

        [Required]
        public string Login { get; set; }

        [Required]
        public string KeyPressedTimes { get; set; }

        [Required]
        public string BetweenKeysTimes { get; set; }

        [Required]
        public string Password { get; set; }

        [Required]
        public string Salt { get; set; }
    }
}

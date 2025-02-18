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
        public string FirstH { get; set; }

        [Required]
        public string FirstUD { get; set; }

        [Required]
        public string SecondH { get; set; }

        [Required]
        public string SecondUD { get; set; }

        [Required]
        public string ThirdH { get; set; }

        [Required]
        public string ThirdUD { get; set; }

        [Required]
        public string Password { get; set; }

        [Required]
        public string Salt { get; set; }
    }
}

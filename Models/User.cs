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
        public string KeyPressedTimesFirst { get; set; }

        [Required]
        public string BetweenKeysTimesFirst { get; set; }

        [Required]
        public string BetweenKeysPressTimesFirst { get; set; }

        [Required]
        public string KeyPressedTimesSecond { get; set; }

        [Required]
        public string BetweenKeysTimesSecond { get; set; }

        [Required]
        public string BetweenKeysPressTimesSecond { get; set; }

        [Required]
        public string KeyPressedTimesThird { get; set; }

        [Required]
        public string BetweenKeysTimesThird { get; set; }

        [Required]
        public string BetweenKeysPressTimesThird { get; set; }

        [Required]
        public string Password { get; set; }

        [Required]
        public string Salt { get; set; }
    }
}

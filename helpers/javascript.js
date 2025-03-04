// Make important rows in Gmail stand out
document.addEventListener("DOMContentLoaded", function (event) {
  setTimeout(function () {
    var imp = document.getElementsByClassName("xg");
    for (var i = 0; i < imp.length; i++) {
      row = imp[i].parentElement.parentElement;
      row.style.background = "#d200dd";
      row.style.fontWeight = "bold";
    }
  }, 2000);
});

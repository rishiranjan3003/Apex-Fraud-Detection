<?php
$servername = "localhost";
$username = "id18905223_apex";
$password = "RishiH@rsh12";
$dbname = "id18905223_apex_fraud_detection";

$e=$_POST["eid"];
$z;



// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT email FROM Apex";
$result = $conn->query($sql);


if ($result->num_rows > 0) {
  // output data of each row
  while($row = $result->fetch_assoc()) {
      $z=$row["email"];
  }
}
    
if(strcmp($z,$e)==0)
{
echo "You have fraud with the accuracy of 0.009786";

}
else
{
echo "You have not fraud";
}


$conn->close();
?>
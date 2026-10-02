ThisBuild / organization := "org.distributeddocintelligence"
ThisBuild / version := "0.1.0"
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "distributed-document-intelligence-spark",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.3" % "provided",
      "org.apache.tika" % "tika-core" % "2.9.2",
      "org.apache.tika" % "tika-parsers-standard-package" % "2.9.2"
    )
  )

import 'package:flutter/material.dart';

void main() {
  runApp(PlantDiseaseApp());
}

class PlantDiseaseApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plant Disease Detection',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: HomePage(),
    );
  }
}
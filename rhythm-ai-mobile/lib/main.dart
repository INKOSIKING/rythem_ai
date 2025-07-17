import 'package:flutter/material.dart';
import 'package:rhythmai_sdk/rhythmai_sdk.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  final ai = RhythmAI(apiKey: "YOUR_API_KEY");
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rhythm AI',
      home: Scaffold(
        appBar: AppBar(title: Text('Rhythm AI')),
        body: ChatScreen(ai: ai),
      ),
    );
  }
}

class ChatScreen extends StatefulWidget {
  final RhythmAI ai;
  ChatScreen({required this.ai});
  @override
  _ChatScreenState createState() => _ChatScreenState();
}
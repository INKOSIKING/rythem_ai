import 'package:flutter/material.dart';
class ChatScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Rhythm AI Chat')),
      body: Center(child: Text('Chat with Rhythm AI!')),
    );
  }
}
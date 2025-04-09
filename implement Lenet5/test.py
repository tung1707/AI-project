with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the 10000 test images: {} %".format(100 * correct/total))